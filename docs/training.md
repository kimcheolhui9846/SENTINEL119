# Zoro-1 SFT 실행 안내

119 통화 전체의 **출동유형 4종·긴급도 3종**을 Zoro의 typed-decision 선택지 손실로 학습한다. 생성형 답변이나 JSON completion을 학습하는 방식은 아니다. 데이터 준비, head-only 지도 적응, LoRA+decision head SFT, 독립 평가 명령을 제공한다.

## 1. 확인한 모델과 데이터

2026-11-19에 [Zoro-1 모델 카드](https://huggingface.co/trillionlabs/Zoro-1)와 인증 후 받은 실제 설정·토크나이저·head·Python 코드를 확인했다.

| 항목 | 고정값·확인 결과 |
| --- | --- |
| 모델 | `trillionlabs/Zoro-1` |
| 모델 revision | `bf1b76b9b4df03b05fcbb73444e1cd4db0f566da` |
| 구조 | `Qwen3_5ForConditionalGeneration`, text hidden size 2,560 |
| head | 기존 `decision_head.pt`; 각 선택지와 decide 위치의 hidden state를 비교하는 pointer head |
| 의존성 | Python 3.11, Transformers 5.12.1, PEFT 0.21.0; 전체 해석 결과는 `uv.lock` |
| 참고 upstream | [trida-stack](https://github.com/trillion-labs/trida-stack/tree/0ddcf516e458a82892d7fe0d0dabeb113d3cf424) |

현재 모델에 포함된 `zoro_decide.py`는 native Qwen3.5와 Transformers ≥5.2를 요구하며 config의 생성 버전은 5.12.1이다. 확인 당시 trida-stack 루트의 Transformers 4.x 안내와 다르므로 **이 저장소의 고정 의존성**을 사용한다. 원격 Python 코드 실행이나 diffusion training kernel은 필요하지 않다.

로컬 [AI Hub 71768](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71768&topMenu=100)의 라벨 ZIP 24개를 검사했다. 아래는 확보된 로컬 파일에 대한 결과이며 전체 공개 데이터셋을 모두 확보했다는 뜻은 아니다.

| 검사 | 결과 |
| --- | ---: |
| JSON | 143,075 |
| 화자 코드 오류 제외 | 4 |
| 빈 발화 제외 | 1 |
| 유효 기록 | 143,070 |
| 발화 시각 순서로 재정렬 | 516 |
| 공식 Training/Validation 사이 중복 제외 | 1그룹, 2건 |
| 최종 사용 가능 그룹 | Training 127,172 / Validation 15,896 |
| 유효 기록 중 비어 있지 않은 symptom / triage | 88,938 / 93,437 |

`symptom`·`triage`의 내용 타당성은 검토하지 않았고 이번 입력·정답에 사용하지 않는다. 공식 화자 정의는 `0=수보자, 1=신고자`지만 역할 라벨의 전수 정확성은 미검증이다. 입력에는 `화자0`, `화자1`이라는 원본 코드만 표시한다. 시각은 원본 밀리초 값으로 보존한다.

## 2. 설치

프로젝트 루트에서 실행한다. GPU 환경에서도 Hugging Face 승인을 받은 계정으로 따로 인증해야 한다. 토큰을 코드·명령행 인자·Git에 넣지 않는다.

```bash
uv sync --locked --extra train --extra test
uv run --no-sync hf auth login
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

실제 학습 기본값은 CUDA BF16이다. CUDA가 없으면 모델 다운로드 전에 실패한다. 설치된 PyTorch의 CUDA 지원과 GPU 드라이버 호환성을 실행 환경에서 확인한다. `--device cpu`도 있지만 전체 모델 FP32 로드는 큰 메모리와 시간이 필요하므로 일반 smoke 경로로 권장하지 않는다. 최초 GPU 실행은 고정 revision의 전체 모델 가중치를 다운로드한다.

### GPU 드라이버에 맞춘 PyTorch 설치

GPU 드라이버가 지원하는 CUDA wheel을 선택한다. 아래는 CUDA 12.6용 PyTorch 2.8.0을 사용하는 예시이며, Windows에서 생성한 `uv.lock`의 PyTorch 2.15와 구별한다. 실행할 GPU와 드라이버의 호환성을 먼저 확인한다.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
mkdir -p artifacts/sft
printf 'torch==2.8.0+cu126\n' > artifacts/sft/cuda-constraints.txt
uv pip install --python .venv/bin/python -e '.[train,test]' -c artifacts/sft/cuda-constraints.txt
uv pip freeze --python .venv/bin/python > artifacts/sft/remote-requirements.txt
```

이 환경에서는 `.venv/bin/python -m sentinel119.train ...`을 사용한다. `uv sync`로 다시 동기화하면 GPU용 PyTorch 선택이 바뀔 수 있다. 실제 설치 버전은 실행별 freeze 파일과 `run.json`에 보관한다. 선택적 FLA/causal-conv1d 커널이 없으면 Transformers의 PyTorch fallback을 사용한다. 해당 환경에서 아래 smoke 실행으로 실제 동작과 실행 시간을 확인한다.

## 3. 데이터 준비

`--data-root`는 Training/Validation 라벨 ZIP이 있는 상위 폴더다. 압축을 풀거나 원본을 수정하지 않는다. 음성 ZIP은 읽지 않는다.

```bash
# 최초 100통화: train 80 / dev 10 / test 10
uv run --no-sync sentinel119-prepare --data-root "<AI Hub 71768 루트>" --output data/processed/aihub119-sft-smoke

# 본 실험: train 2,000 / dev 300 / test 500
uv run --no-sync sentinel119-prepare --data-root "<AI Hub 71768 루트>" --output data/processed/aihub119-sft --train-count 2000 --dev-count 300 --test-count 500
```

생성물과 원본은 Git에서 제외된다. 이미 내용이 있는 출력 폴더에는 덮어쓰지 않으므로 재실행 시 새 경로를 지정한다.

분할 규칙:

- 전체 확보 라벨을 검사한 후 `recordId`와 정규화된 전체 대화 텍스트 해시의 연결 그룹을 만든다. 연결 관계는 전이적으로 묶는다. 의미가 비슷한 대화의 중복 탐지는 포함하지 않는다.
- 공식 Training/Validation에 걸친 그룹과 라벨이 충돌하는 그룹은 모두 제외한다. 나머지 그룹마다 대표 통화 하나를 선택한다.
- 공식 Training에서 train/dev를, 공식 Validation에서 test를 뽑는다. 유형·긴급도 조합으로 층화하며 작은 smoke 세트에서도 가능한 경우 각 유형·긴급도를 확보한다. seed 기본값은 119다.
- 모델 입력 `state`에는 시각순 대화만 둔다. 정답은 `questions.*.label`, 원본 ID·턴·시각·출처는 `_meta`에 둔다. 전체 주소·증상 등 통화 메타데이터는 입력에 포함하지 않는다.
- **최종 통화 라벨을 prefix에 복사하지 않는다.** 현재 코드는 full-call 지도학습만 받는다. 조기 COMMIT/WAIT, 자원·병원 추천은 별도 데이터와 검증이 필요하다.

출력은 `train.jsonl`, `dev.jsonl`, `test.jsonl`, `manifest.json`이다. manifest에 수량, 제외 사유, 분포, 누락 클래스와 파일 SHA-256을 기록한다. 생성 파일에는 신고 원문이 있으므로 공유 문서나 Git에 추가하지 않는다.

## 4. 학습

먼저 GPU에서 기존 head의 backward·저장·재로드를 2 step 확인한다.

```bash
uv run --no-sync sentinel119-train --data data/processed/aihub119-sft-smoke --output runs/zoro-head-smoke --mode head-only --max-steps 2 --grad-accum 1 --eval-every 1
```

`head-only`는 백본을 고정하고 기존 pointer head만 업데이트하는 **head-only 지도 적응**이다. LoRA나 전체 모델 SFT라고 부르지 않는다.

다음으로 LoRA 경로를 같은 작은 데이터에서 확인한다.

```bash
uv run --no-sync sentinel119-train --data data/processed/aihub119-sft-smoke --output runs/zoro-lora-smoke --mode lora --max-steps 2 --grad-accum 1 --eval-every 1
```

LoRA는 **텍스트 full-attention의 q_proj/v_proj**에만 rank 8 / alpha 16으로 적용한다. 기존 decision head도 함께 학습한다. gated-delta 층, vision tower, 원본 가중치는 고정한다. non-reentrant gradient checkpointing을 사용한다. 이는 전체 attention/MLP를 학습하는 LoRA나 full fine-tuning이 아니다.

GPU smoke가 성공하면 본 실험을 실행한다. 다음은 2,000통화를 한 번씩 사용하는 500 optimizer step 예시이며 검증된 최적 설정은 아니다.

```bash
uv run --no-sync sentinel119-train --data data/processed/aihub119-sft --output runs/zoro-lora-v1 --mode lora --max-steps 500 --grad-accum 4 --eval-every 100 --learning-rate 0.0001
```

한 microbatch는 한 통화의 두 질문이며 질문마다 독립적인 causal row를 사용한다. 두 질문의 CE 평균을 통화 손실로 삼고 선택지 순서를 학습 때마다 섞는다. 서로 다른 통화를 한 token sequence에 이어 붙이지 않는다.

기본 row 길이 제한은 4,096이다. 긴 통화를 자른 채 최종 라벨로 학습하지 않고 해당 통화를 제외해 coverage에 기록한다. `--max-length`는 최대 8,192, state는 최대 7,552 token이다. 제외된 통화가 있으면 수량과 클래스 분포를 함께 확인하고 길이·메모리 예산을 조정한다.

학습 산출물:

| 파일 | 내용 |
| --- | --- |
| `decision_head.pt` | dev 평균 NLL이 가장 낮은 학습 step의 head |
| `adapter/` | LoRA 모드의 PEFT adapter; 같은 step의 head와 함께 사용 |
| `temperature.json` | `text: 1.0`, 119 calibration 미수행 |
| `history.jsonl` | step별 손실·gradient norm, 주기별 dev 평가 |
| `run.json` | 모델·참고 upstream revision, 패키지 버전, 학습 대상, 데이터 manifest, 제외 통화, Base/SFT dev 수치, gradient·재로드 검증, 완료/실패 상태 |

평가는 정확도, 고정 클래스 전체 macro-F1, confusion matrix, 긴급도 하향 오판률, NLL을 기록한다. 없는 클래스의 F1도 0으로 포함하고 누락을 표시한다. Base와 적응 모델 모두 raw temperature 1.0으로 비교한다. 원본 모델 카드의 calibration 수치를 119에 그대로 적용하지 않는다. SFT가 Base보다 나빠도 결과를 숨기지 않으며 `improved_dev_nll`을 확인한다.

`--checkpoint`는 평가용이며 optimizer resume 기능은 없다. 출력 폴더마다 독립 실행한다. 유한 step 후 자동 종료하며 인스턴스를 생성·증설하지 않는다.

## 5. 독립 평가와 모델 사용

모델·설정 선택은 dev로 끝낸 후 test를 별도로 평가한다. 아래 두 결과를 같은 데이터·길이 설정으로 비교한다.

```bash
uv run --no-sync sentinel119-train --data data/processed/aihub119-sft --output runs/zoro-base-test --evaluate-only test
uv run --no-sync sentinel119-train --data data/processed/aihub119-sft --output runs/zoro-sft-test --evaluate-only test --checkpoint runs/zoro-lora-v1
```

checkpoint의 mode는 저장된 `run.json`에서 읽는다. 고정 모델 revision, base head/config 해시와 데이터 split manifest가 다르면 실패한다. `--model-dir`로 로컬 모델을 지정한 경우 revision 미검증 override로 명시하고 같은 로컬 경로·config/head 해시를 요구한다. 로컬 override의 대용량 백본 파일 해시는 검증하지 않는다.

추론 통합 시 `sentinel119.model.load_zoro(head_path=..., adapter_path=...)`로 base+head+adapter를 함께 로드하고 `sentinel119.decision.encode`로 입력을 만든다. head-only에는 `adapter_path`를 주지 않는다. `model.eval()`과 `torch.no_grad()`에서 질문별 logits에 softmax를 적용한다. 저장 디렉터리는 전체 Hugging Face 모델이 아니므로 그대로 원본 Zoro 서버의 모델 경로로 넘기지 않는다. 자원·병원 태스크는 이 분류용 adapter로 검증하지 않았으므로 원본 모델 경로를 유지한다.

## 6. 검증 절차

```bash
uv run --no-sync pytest -q
```

자동 검사는 데이터 누출 방지, 전이적 중복 그룹, 충돌 라벨, 층화 분할, token/label 분리, overflow 거절, frozen 가중치 보존, tiny native Qwen3.5의 mixed-attention forward/backward, head/LoRA 저장·재로드를 확인한다. 테스트 통화는 합성이다. CLI 통합 검사는 작은 로컬 모델의 학습과 독립 평가 명령을 실행한다.

`tools/check_zoro_contract.py`는 고정 revision의 공식 Python 파일과 실제 tokenizer/head를 사용해 인코딩·pointer 계산의 일치와 head gradient·저장·재로드를 검사한다. 검토한 reference directory와 생성 데이터 경로를 명시한다.

```bash
uv run --no-sync python tools/check_zoro_contract.py --reference-dir tmp/zoro-reference --data data/processed/aihub119-sft-smoke --output artifacts/sft/zoro-contract.json
```

이 계약 검사의 head gradient 입력은 합성 hidden state다. 실제 GPU 학습 검증은 head-only와 LoRA smoke의 `run.json`에서 완료 상태, gradient, frozen base, 저장·재로딩 결과를 확인한다. 본 학습에서는 coverage와 dev 선택 기준을 확인하고, 선택을 마친 후 독립 test 결과를 비교한다.

NLL이나 macro-F1 개선만으로 긴급도 판단의 적합성을 결론 내리지 않는다. 클래스별 recall, 긴급도 하향 오판, 누락 클래스도 함께 검토한다. 전체 통화 분류 결과는 prefix 조기 판단·자원/병원 추천의 검증을 대신하지 않는다. 각 실행의 가중치와 원본 평가 기록은 Git 제외 경로에 보관한다.

## 7. 전체 통화 전사문을 결합하는 실험

같은 통화에 대해 고정된 Zoro SFT의 선택지 점수와 문자 2–5그램 TF-IDF 로지스틱 회귀의 로그 점수를 결합한다. 전사문 분류기는 공식 Training에서 기존 dev 그룹을 제외한 4만 건으로 학습한다. 입력에는 화자별 발화만 사용한다. 이 경로는 완료된 통화의 데이터셋 라벨을 예측하는 실험용이다.

CPU 환경에는 선택 의존성 `transcript`를 설치한다. GPU에서 점수 추출에는 기존 `train` 환경과 캐시된 고정 revision을 사용한다. 다음 명령은 프로젝트 루트에서 실행하며 출력마다 새 경로가 필요하다. 서로 다른 머신을 쓸 때는 추출한 점수 파일을 동일한 상대 경로에 둔다.

```bash
UV_PROJECT_ENVIRONMENT=.venv-transcript uv sync --locked --extra transcript
CPU=.venv-transcript/bin/python
GPU=.venv/bin/python
DATA=data/processed/aihub119-sft
OUT=artifacts/sft/full-call-comparison
CKPT=runs/zoro-lora-v1
"$CPU" tools/fit_transcript_model.py --data-root "<AI Hub 71768 루트>" --data "$DATA" --output "$OUT/model"
"$GPU" tools/export_decision_scores.py --checkpoint "$CKPT" --split dev --data "$DATA" --output "$OUT/sft-dev.json"
"$CPU" tools/evaluate_transcript_model.py --data "$DATA" --model "$OUT/model" --scores "$OUT/sft-dev.json" --output "$OUT/dev"
"$GPU" tools/export_decision_scores.py --checkpoint "$CKPT" --split test --data "$DATA" --output "$OUT/sft-test.json"
"$CPU" tools/evaluate_transcript_model.py --data "$DATA" --model "$OUT/model" --scores "$OUT/sft-test.json" --dev-policy "$OUT/dev/policy.json" --output "$OUT/test"
```

결합 비중은 dev에서 태스크별로 선택한다. 출동유형은 정확도 하락을 1%p까지 허용하고, 긴급도는 기존 정확도 이상이면서 하향 오판이 20% 이상 줄고 상→하 건수가 증가하지 않는 후보를 비교한다. 이 실험 조건을 충족하는 후보 중 macro-F1이 가장 높은 설정을 저장하며, 기존 F1에 미달하면 기존 결정을 사용한다. 테스트는 저장된 설정으로만 실행한다. `metrics.json`에는 Zoro 단독·전사문 단독·결합 모델의 혼동행렬이 들어간다.

단일 통화 추론에는 `sentinel119.transcript.predict_full_call`에 전사문, 같은 revision·head·adapter를 로드한 모델과 tokenizer, `np.load("…/transcript.npz", allow_pickle=False)`로 연 가중치, 읽어 들인 `policy.json`을 전달한다. 정책 파일에는 대응하는 가중치의 SHA-256과 라벨 순서가 있다. 출력은 출동유형과 긴급도 라벨이다.

## 8. 지침 기반 선택지 문구와 학습 목적 옵션

모든 옵션의 기본값은 기존 동작(선택지 이름만, 일반 CE, 고정 학습률, dev NLL 선택)과 같다. 새 값은 실행별 `run.json`의 `config`·`prompt`·`objective`에 기록된다.

### 선택지 문구 (`--prompt-variant`)

| 값 | 내용 |
| --- | --- |
| `bare` | 준비된 데이터의 선택지 이름만 사용 |
| `protocol` | 긴급도 `high` 선택지에 「수보단계 중증응급환자 인지 프로토콜」(2023 표준지침, `p2-s2-c02`, p.156–159)의 조기 긴급출동 상황과 특별구급대 출동 대상 범주를 덧붙임 |
| `protocol-checklist` | `protocol`에 같은 장의 확인 질문을 긴급도 질문 문구 뒤에 추가 |

- 지침 지식베이스는 이 저장소에 넣지 않는다. `--guideline-dir`로 `extracted/chapters`를 포함한 외부 지식 루트를 지정한다. 학습 인스턴스에는 해당 디렉터리를 복사해 사용할 수 있다. 렌더링은 장의 형식이 예상과 다르면 실패한다.
- 기록되는 것은 렌더링된 문구, 원본 장 파일 이름·SHA-256·쪽이다. 체크포인트 평가와 `tools/export_decision_scores.py`는 기록된 문구를 재사용하므로 지식 디렉터리가 필요 없다. 다른 variant를 지정하면 실패한다.
- 프로토콜은 `중`·`하`를 정의하지 않으므로 해당 선택지는 이름만 유지한다. 프로토콜 범주와 AI Hub `상` 라벨의 일치는 **미검증 가설**이며, dev 비교로 확인한다.
- 단일 통화 추론에서 `predict_full_call(..., prompt=run["prompt"])`로 같은 문구를 전달한다.

### 학습 목적·최적화

| 옵션 | 의미 |
| --- | --- |
| `--urgency-weight` | 두 질문 손실의 가중 평균에서 긴급도 비중 (출동유형 1) |
| `--urgency-label-smoothing` | 긴급도 CE의 label smoothing |
| `--urgency-cost-weight`, `--downward-cost` | 긴급도에 예측분포 기준 비대칭 순서 비용 기대값을 더한다. 상향 오판은 단계당 1, 하향 오판은 단계당 `downward-cost` |
| `--dispatch-logit-adjust` | 학습 시에만 출동유형 logits에 τ·log(train 사전확률)을 더한다. 평가·추론은 원래 logits |
| `--oversample` | 출동유형 라벨 반복 표집, 예 `other=3,rescue=3`. logit adjustment와 동시 사용은 보정이 중복되므로 피한다 |
| `--lora-targets` | `attention-qv`(기존), `attention`(q/k/v/o), `attention-mlp`(q/k/v/o + 전 층 MLP). gated-delta 투영과 vision은 고정 |
| `--head-learning-rate`, `--warmup-ratio`, `--schedule cosine` | head 학습률 분리, 선형 warmup 후 cosine 감소 |
| `--select-by composite`, `--max-downward-rate` | dev 두 macro-F1 평균으로 선택하되 하향 오판율 상한을 만족하는 step을 우선. 만족하는 step이 없으면 가장 높은 F1 평균 step이며 `result.selection.feasible=false`로 기록 |
| `--keep-checkpoints` | 평가한 모든 step을 `checkpoints/step-N/`에 보존 |

dev NLL은 학습 목적과 관계없이 일반 CE로 계산한다. `history.jsonl`의 `train_loss`는 선택한 학습 목적의 값이다.

### 지침과 학습 목적을 바꾸는 후속 실험 예시

```bash
DATA=data/processed/aihub119-sft
KB="<지침 지식 루트>"
# 1) Base 선택지 문구 비교 (dev, 학습 없음)
for v in bare protocol protocol-checklist; do
  .venv/bin/python -m sentinel119.train --data "$DATA" --output runs/base-dev-$v --evaluate-only dev \
    --prompt-variant $v --guideline-dir "$KB"
done
# 2) 2k 통화, 새 설정. 2,000 = 125 step × 16 통화
.venv/bin/python -m sentinel119.train --data "$DATA" --output runs/zoro-lora-v2a --mode lora \
  --prompt-variant <1에서 고른 값> --guideline-dir "$KB" \
  --lora-targets attention-mlp --lora-rank 16 --lora-alpha 32 \
  --max-steps 125 --grad-accum 16 --eval-every 25 --learning-rate 1e-4 --head-learning-rate 3e-4 \
  --warmup-ratio 0.05 --schedule cosine --urgency-weight 1.5 --urgency-label-smoothing 0.1 \
  --urgency-cost-weight 0.3 --downward-cost 2 --dispatch-logit-adjust 1.0 \
  --select-by composite --max-downward-rate 0.2 --keep-checkpoints
```

1단계는 dev에서만 고른다. 기존 test 500건은 여러 번 확인했으므로 최종 비교에는 새로 예약한 독립 평가 그룹을 사용한다. 하향 오판 상한 0.2는 기존 dev의 Base 10.7%, SFT 29.7%, 결합 18.3% 사이에서 정한 실험값이며 임상 허용 기준이 아니다.

## 9. 2천 통화 헤드 단독 대조와 8천 통화 확대

첫 대조에서는 기존 LoRA와 같은 2,000통화, seed 119, 500 step, 누적 4통화, 학습률 1e-4, 일반 CE, `bare` 문구를 쓴다. 원본 Zoro의 판단 헤드만 업데이트한다. 100 step마다 가중치를 보관하고 기존 LoRA와 동일하게 dev 평균 NLL이 가장 낮은 step을 비교한다. 8절의 지침·표집·비용 옵션은 별도 실험이다.

```bash
DATA=data/processed/aihub119-sft
NEXT=data/processed/aihub119-head-comparison
BASELINE=runs/zoro-lora-v1
HEAD=runs/zoro-head-control-2k

# 기존 GPU 환경에 현재 소스와 원래 데이터·체크포인트가 있을 때 실행한다.
# check는 데이터 해시와 기존 학습 설정을 확인한다. GPU 모델을 로드하는 동작은 train에서 시작한다.
.venv/bin/python tools/head_control.py --data "$DATA" --baseline "$BASELINE" --output "$HEAD" --action check
.venv/bin/python tools/head_control.py --data "$DATA" --baseline "$BASELINE" --output "$HEAD" --action train
```

학습이 끝나면 `$HEAD/head-comparison.json`에 각 모델의 선택 step, dev 출동유형·긴급도 macro-F1, 하향 오판율, 상→하 건수, 기타 정답·표본 수와 차이가 기록된다. 완료한 실행의 비교만 다시 만들려면 `--action compare`를 사용한다.

### 평가셋 예약과 확대 기준

원본 ZIP이 있는 머신에서 다음 명령으로 8천 건과 평가용 그룹을 함께 준비한다. 기존 TF-IDF의 `fit.json`으로 4만 학습 그룹을 재현하고 해시를 검증한다. 기존 데이터 폴더는 첫 대조에 계속 사용한다.

```bash
.venv/bin/python -m sentinel119.prepare_comparison --data-root "<AI Hub 71768 루트>" \
  --reference "$DATA" --transcript-fit artifacts/sft/hybrid-v2/transcript-40k/fit.json --output "$NEXT"
```

| 파일 | 건수 | 용도 |
| --- | ---: | --- |
| train.jsonl | 8,000 | 기존 2,000과 공식 Training의 추가 6,000 |
| dev.jsonl | 300 | 기존 dev와 동일한 통화·순서로 학습 중 점검 |
| confirm.jsonl | 1,000 | 새 비교용 dev. 기존 Zoro와 TF-IDF 학습 그룹, 기존 dev, 확대 학습 그룹 모두 제외 |
| test.jsonl | 1,000 | 공식 Validation에서 기존 test 500그룹을 제외하고 예약한 최종 평가 |

두 2천 건 모델의 설정을 고정한 후 같은 `confirm`에서 비교한다. 실험용 확대 기준으로 **헤드 단독의 두 macro-F1이 각각 LoRA 대비 0.02 이내로 낮거나 더 높고, 하향 오판율과 상→하 건수는 증가하지 않는 경우**를 제안한다. 기타 정답 수와 상향 오판도 혼동행렬에서 함께 확인한다. 이 기준은 한 표본에서 다음 GPU 실험을 고르는 용도다.

```bash
for model in "$BASELINE" "$HEAD"; do
  .venv/bin/python -m sentinel119.train --data "$NEXT" --reference-data "$DATA" --checkpoint "$model" \
    --evaluate-only confirm --output "${model}-confirm" --local-files-only
done
# 헤드 단독을 확대한 경우. 원본 Zoro에서 8천 통화를 1회 학습한다.
.venv/bin/python -m sentinel119.train --data "$NEXT" --mode head-only --prompt-variant bare \
  --max-steps 2000 --grad-accum 4 --eval-every 400 --learning-rate 1e-4 --seed 119 \
  --keep-checkpoints --local-files-only --output runs/zoro-head-8k
```

최종 후보를 결정한 뒤에만 새 `test`를 평가한다. 8천 건 체크포인트는 `--data "$NEXT" --evaluate-only test`로 평가한다. 기존 데이터에서 학습한 체크포인트는 `--reference-data "$DATA"`도 전달한다. 평가기는 부모 manifest의 일치를 확인하고 그룹·녹취 ID·정규화 전사문 해시가 부모 train/dev/test와 겹치면 중단한다.
