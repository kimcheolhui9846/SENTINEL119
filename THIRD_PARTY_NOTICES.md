# Third-party notices

`src/sentinel119/decision.py` follows the text-choice delimiter and escaping protocol
in [trida-stack](https://github.com/trillion-labs/trida-stack), revision
`0ddcf516e458a82892d7fe0d0dabeb113d3cf424`, and the released Zoro-1 `decide_request.py`.
`src/sentinel119/model.py` adapts its pointer-head formula and parameter layout.
Modified here for strict full-call length checks, supervised labels, a frozen
backbone, and local checkpoint verification.

- Copyright 2026 Trillion Labs.
- The original pointer head and decision encoding derive from
  [Kev](https://github.com/jaredpalmer/kev), copyright Jared Palmer and the Kev contributors.
- These portions are licensed under Apache License 2.0; see
  [the included license](third_party/Apache-2.0.txt).

Model weights and AI Hub records are downloaded separately and are not distributed
by this repository. The block-diffusion training kernels are not included or used.
