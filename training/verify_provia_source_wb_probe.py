"""Independent arithmetic verification of recorded F24 metadata; no media reads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    diag = ROOT / "diagnostics"
    path = diag / "provia-source-wb-probe.json"
    report = json.loads(path.read_text())
    archive = {r["name"]: r for r in json.loads((diag / "cn-step1-wb-reference-audit.json").read_text())["scenes"]}
    manifest_path = diag / "provia-source-wb-probe-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert digest(manifest_path) == report["manifest_sha256"]
    assert len(report["rows"]) == len(archive) == 24
    assert {r["name"] for r in report["rows"]} == set(archive)
    assert report["crosschecks_passed"]
    auto, groups = [], {}
    for row in report["rows"]:
        old = archive[row["name"]]
        assert row["raf_sha256"] == old["source_raf_sha256_at_probe"]
        assert not row["crosscheck_problems"]
        records = {r["tag"]: r for r in row["records"]}
        assert len(records) == len(row["records"]) == 3
        # Direct indexed conversion; do not import producer channel-order code.
        current = records["0xf00e"]["values"]
        assert [current[1], current[0], current[2]] == old["libraw_cam_mul"][:3]
        assert row["archived_cam_mul_matching_as_shot_tags"] == ["0xf00e"]
        assert row["candidates"]["0x2000"]["scored"] is False
        if row["source_wb_mode"] != 0:
            assert not row["candidates"]["0xf00d"]["scored"]
            continue
        values = records["0xf00d"]["values"]
        got = [values[1] / values[0], values[2] / values[0]]
        expected = [old["provia"]["WBLevelR"] / old["provia"]["WBLevelG"],
                    old["provia"]["WBLevelB"] / old["provia"]["WBLevelG"]]
        assert got == expected
        assert row["candidates"]["0xf00d"]["relative_target_error_RB"] == [0.0, 0.0]
        assert max(abs(a / b - 1) for a, b in zip(got, old["libraw_gain_RB_over_G"])) > .005
        auto.append(row["name"])
        k = str(row["shift_steps"])
        groups[k] = groups.get(k, 0) + 1
    assert len(auto) == 11 and groups == {"[5.0, -6.0]": 7, "[4.0, -5.0]": 4}
    assert report["screens"]["0xf00d"]["passed"]
    assert not report["screens"]["0x2000"]["passed"]
    out = {"passed": True, "code_sha256": digest(Path(__file__)), "report_sha256": digest(path),
           "manifest_sha256": digest(manifest_path), "n_source_as_shot_anchors": 24,
           "n_auto_exact_neutral_target_ratio_matches": len(auto), "shift_counts": groups,
           "limits": "Saved metadata arithmetic only; no fresh media, pixel or applied-gain validation"}
    target = diag / "provia-source-wb-probe-verification.json"
    assert not target.exists()
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
