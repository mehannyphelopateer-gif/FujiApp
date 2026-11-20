"""Post-hoc reporting-unit diagnosis of already-read, train-only C8 WB metadata.

Preserves the frozen probe's INCONCLUSIVE_SCALE outcome. This separate analysis
asks whether one common RGB scale, fixed by green alone, explains its mismatch.
It reads JSON evidence only: no images, no new WB variants, no gain fitting and
no runtime or policy changes. The target values were seen before this analysis
was specified; its result is development evidence, never independent acceptance.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "training/diagnostics"
OUTPUT = DIAG / "cuban-auto-wb-units-diagnosis.json"
BINDINGS = {
    "cuban-auto-wb-predictions.json": "392684b688915bf0dd3501e3a3373742edd4b1e2af77dd283f87cd16c17fb4ee",
    "cuban-auto-wb-metadata-manifest.json": "0f2ff5250e2045f97fb0e228b9d5b7ef4da4415576bd5687973c35b484fbe518",
    "cuban-auto-wb-qualification.json": "6f3228d1707805f63d4268bf0f2e2497d8a97b0047d39ac442dc08e1d52abb5c",
    "cuban-auto-wb-metadata-report.json": "39c283a344d98497268fbe2f5b3304a88f3ca257e981c89f721a063af6948161",
}
VARIANTS = ("H0", "H1", "H2", "H3", "H4")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value):
    require(isinstance(value, list) and len(value) == 3,
            "Expected three RGB levels")
    require(all(type(x) is int and 0 < x <= 2**32 - 1 for x in value),
            "RGB levels must be positive uint32 integers")
    return value


def compare_vector(predicted, observed):
    """Exact homogeneous-vector comparison; no tolerance or per-channel scaling.

    The unique scale is observed G / predicted G. Cross-products avoid floating
    point arithmetic. A changed red or blue channel cannot be hidden by scale.
    """
    p, t = vector(predicted), vector(observed)
    scale = Fraction(t[1], p[1])
    residuals = [t[c] * p[1] - p[c] * t[1] for c in range(3)]
    return {"scale": str(scale), "cross_product_errors_rgb": residuals,
            "rgb_proportional_exact": all(x == 0 for x in residuals)}


def diagnose_rows(rows):
    require(len(rows) == 8 and len({r["name"] for r in rows}) == 8,
            "Expected the eight unique locked C scenes")
    require(all(type(r["capture_mode"]) is int for r in rows), "Invalid WB mode")
    require(sum(r["capture_mode"] == 0 for r in rows) == 3,
            "Expected three source-Auto controls")
    per_variant = {}
    for variant in VARIANTS:
        cases = [{"name": r["name"], "capture_mode": r["capture_mode"],
                  **compare_vector(r["predictions"][variant], r["observed"])}
                 for r in rows]
        scales = sorted({r["scale"] for r in cases})
        same_units = len(scales) == 1
        auto = [r for r in cases if r["capture_mode"] == 0]
        other = [r for r in cases if r["capture_mode"] != 0]
        controls_exact = same_units and all(r["rgb_proportional_exact"] for r in auto)
        per_variant[variant] = {
            "scales": scales, "one_scale_across_all_scenes": same_units,
            "source_auto_rgb_exact": sum(r["rgb_proportional_exact"] for r in auto),
            "source_auto_n": len(auto),
            "source_non_auto_rgb_exact": sum(r["rgb_proportional_exact"] for r in other),
            "source_non_auto_n": len(other),
            "non_auto_interpretable_under_same_variant": controls_exact,
            "cases": cases,
        }
    return per_variant


def main():
    evidence = {}
    for name, expected in BINDINGS.items():
        path = DIAG / name
        require(digest(path) == expected, f"Evidence hash mismatch: {name}")
        evidence[name] = json.loads(path.read_text())
    pred = evidence["cuban-auto-wb-predictions.json"]
    manifest = evidence["cuban-auto-wb-metadata-manifest.json"]
    qual = evidence["cuban-auto-wb-qualification.json"]
    old_path = DIAG / "cuban-auto-wb-metadata-report.json"
    old_sha = digest(old_path)
    old = evidence[old_path.name]
    require(old["result"]["verdict"] == "INCONCLUSIVE_SCALE",
            "Original result must remain INCONCLUSIVE_SCALE")
    require(old["manifest_sha256"] == BINDINGS["cuban-auto-wb-metadata-manifest.json"]
            and old["qualification_sha256"] == BINDINGS["cuban-auto-wb-qualification.json"],
            "Original evidence bindings differ")
    require(qual["passed"] is True and qual["manifest_sha256"] == old["manifest_sha256"],
            "Original qualification did not pass")
    require(pred["variants"] == list(VARIANTS), "Prediction variants changed")
    index = {r["name"]: r for r in pred["scenes"]}
    require(len(index) == len(pred["scenes"]) == 8, "Prediction identity mismatch")
    rows = old["rows"]
    require({r["name"] for r in rows} == set(index)
            == {r["name"] for r in qual["rows"]}, "Evidence scene sets differ")
    # Manifest identities are already bound by the original qualification. Keep
    # its complete JSON hash, without following paths to any TIFF or RAF here.
    require(isinstance(manifest, dict), "Malformed original manifest")
    for row in rows:
        expected = index[row["name"]]
        require(row["predictions"] == expected["predictions"]
                and row["capture_mode"] == expected["capture_mode"],
                "Recorded predictions or source mode changed")
    results = diagnose_rows(rows)
    require(digest(old_path) == old_sha, "Original report changed during diagnosis")
    report = {
        "scope": __doc__.strip(),
        "evidence": {**BINDINGS, old_path.name: old_sha},
        "implementation_sha256": digest(Path(__file__)),
        "original_verdict_preserved": "INCONCLUSIVE_SCALE",
        "method": "One exact common RGB reporting scale, determined by green; all five frozen variants reported; no rounding allowance or per-scene best variant.",
        "per_variant": results,
        "limitations": "Metadata development diagnostic on 3 source-Auto controls and 5 non-Auto captures (4 Kelvin 6600K, 1 White Fluorescent). Tag-as-RGB meaning remains inferred. No held-out evidence, rendered-image parity, general manual-WB neutralisation, runtime support or model selection follows.",
    }
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    for name, result in results.items():
        print(name, "scale", result["scales"], "Auto", result["source_auto_rgb_exact"],
              "/ 3; non-Auto", result["source_non_auto_rgb_exact"], "/ 5; interpretable",
              result["non_auto_interpretable_under_same_variant"])


if __name__ == "__main__":
    main()
