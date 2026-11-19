import json
import zipfile

import pytest

from sentinel119.prepare import digest, eligible_groups, prepare, scan, stratified_take
from sentinel119.prepare_comparison import prepare_comparison
from sentinel119.train import evaluation_split, load_split, write_json
from test_prepare import call


def test_reserved_data_excludes_prior_exposure_and_requires_matching_parent(tmp_path):
    root, ref, output = (tmp_path / n for n in ("source", "reference", "expanded"))
    root.mkdir()
    for prefix in ("TL", "VL"):
        with zipfile.ZipFile(root / f"{prefix}_synthetic.zip", "w") as z:
            for i in range(65):
                label = ["구급", "기타", "구조", "화재"][i % 4]
                urgency = ["상", "중", "하"][i % 3]
                z.writestr(f"{i}.json", json.dumps(call(f"{prefix}-{i}", f"합성 {prefix} {i}", label, urgency)))
    base = prepare(root, ref, train_count=12, dev_count=4, test_count=4)
    entries, groups, _ = scan(root)
    pools, _ = eligible_groups(entries, groups)
    _, rest = stratified_take(pools["Training"], 4, 119)
    lexical, _ = stratified_take(rest, 30, 120)
    lexical_ids = {g for g, _ in lexical}
    fit = {"train_count": 30, "train_groups_sha256": digest("\n".join(sorted(lexical_ids))),
           "split_hashes": {k: v["sha256"] for k, v in base["splits"].items()}}
    fit_path = tmp_path / "fit.json"
    write_json(fit_path, fit)
    new = prepare_comparison(root, ref, fit_path, output, train_count=24, reserve_count=6)
    prior = {"dataset_manifest": base}

    def ids(dataset, name):
        return {r["_meta"]["group_id"] for r in load_split(dataset, name)[0]}

    assert new["splits"]["dev"] == base["splits"]["dev"]
    assert ids(ref, "train") < ids(output, "train")
    assert ids(output, "train").isdisjoint(ids(output, "confirm"))
    assert lexical_ids.isdisjoint(ids(output, "confirm"))
    for name in ("confirm", "test"):
        rows, _ = evaluation_split(output, name, prior, ref)
        assert len(rows) == 6
        with pytest.raises(ValueError, match="manifest"):
            evaluation_split(output, name, prior)
    with pytest.raises(ValueError, match="parent"):
        evaluation_split(output, "dev", prior, ref)
    # A copied parent test split must still be rejected even with a consistent local hash.
    (output / "test.jsonl").write_bytes((ref / "test.jsonl").read_bytes())
    new["splits"]["test"] = base["splits"]["test"]
    write_json(output / "manifest.json", new)
    with pytest.raises(ValueError, match="overlaps parent"):
        evaluation_split(output, "test", prior, ref)
