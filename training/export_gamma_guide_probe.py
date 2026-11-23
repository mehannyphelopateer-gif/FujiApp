"""Synthetic-only native/browser feasibility of the fixed encoded guide.

No photos, target caches or trained checkpoints are read. Uses the frozen
Round11 export normalisation conversion and unchanged browser verifier.
This is feasibility, never model selection or an accuracy measurement.
"""
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export_round11_onnx import replace_export_group_norms
from gamma_encoded_guide import GammaEncodedGuideHybrid, GUIDE_CONTRACT, TABLE_SHA256
from matched_gamma_data import ROOT, CKPT, write_once_json, sha

OUT = CKPT / "gamma-guide-export-probe"


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, linear_rgb, capture_dr_condition):
        return self.model(linear_rgb, condition=capture_dr_condition)[0]


def main():
    if OUT.exists():
        raise ValueError("Refusing existing probe directory")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = GammaEncodedGuideHybrid(condition_channels=1).eval()
    # The trained refinement will be nonzero. Exercise it with reproducible
    # synthetic weights instead of relying only on its zero initial output.
    with torch.no_grad():
        model.refinement.out_conv.weight.normal_(0, .002)
        model.refinement.out_conv.bias.normal_(0, .001)
    export_model = copy.deepcopy(model)
    norms = replace_export_group_norms(export_model)
    reference = Wrapper(model)
    wrapper = Wrapper(export_model).eval()
    rng = np.random.default_rng(20261123)
    cases = [
        ("odd-landscape-dr100", (1, 3, 73, 111), 0.),
        ("odd-portrait-dr200", (1, 3, 111, 73), .5),
        ("working-landscape-dr400", (1, 3, 1026, 1536), 1.),
        ("working-portrait-dr100", (1, 3, 1536, 1026), 0.),
    ]
    inputs = [(name, rng.integers(0,65536,size=shape,dtype=np.uint16).astype(np.float32)/65535., cond)
              for name,shape,cond in cases]
    OUT.mkdir()
    model_name = "synthetic-gamma-guide.onnx"
    path = OUT / model_name
    with torch.inference_mode():
        torch.onnx.export(wrapper,(torch.from_numpy(inputs[0][1]),torch.tensor([[0.]])),path,
                          opset_version=20,dynamo=False,external_data=False,
                          input_names=["linear_rgb","capture_dr_condition"],
                          output_names=["corrected_linear_rgb"],
                          dynamic_axes={"linear_rgb":{2:"height",3:"width"},
                                        "corrected_linear_rgb":{2:"height",3:"width"}})
    onnx.checker.check_model(str(path),full_check=True)
    session = ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
    vectors, results = [], []
    for name,image,cond in inputs:
        condition = np.array([[cond]],dtype=np.float32)
        with torch.inference_mode():
            expected = reference(torch.from_numpy(image),torch.from_numpy(condition)).numpy()
        actual = session.run(None,{"linear_rgb":image,"capture_dr_condition":condition})[0]
        diff = np.abs(actual-expected)
        ok = bool(np.isfinite(diff).all() and diff.max()<=1e-4 and diff.mean()<=1e-5)
        input_name,expected_name = name+"-input.f32",name+"-expected.f32"
        image.astype('<f4').tofile(OUT/input_name)
        expected.astype('<f4').tofile(OUT/expected_name)
        vectors.append({"name":name,"shape":list(image.shape),"capture_dr":int(100*2**(2*cond)),
                        "condition":cond,"input":input_name,"source":"synthetic"})
        result = {"name":name,"expected":expected_name,"max_abs":float(diff.max()),
                  "mean_abs":float(diff.mean()),"passed":ok}
        results.append(result)
        print(json.dumps(result),flush=True)
    report = {"format_version":1,"opset":20,"scope":"Synthetic export feasibility only; no photos, trained checkpoint, fitting or accuracy claim",
              "guide_contract":GUIDE_CONTRACT,"table_sha256":TABLE_SHA256,
              "native_tolerances":{"max_abs":1e-4,"mean_abs":1e-5},
              "versions":{"torch":torch.__version__,"onnx":onnx.__version__,"onnxruntime":ort.__version__},
              "bindings":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'training/gamma_encoded_guide.py',
                           ROOT/'training/model.py',ROOT/'training/eval_provia_gamma_length.py',
                           ROOT/'training/export_round11_onnx.py',ROOT/'scripts/verify-round11-onnx-web.mjs']},
              "cases":vectors,"models":[{"seed":42,"model":model_name,"model_sha256":sha(path),
                      "export_group_norm_replacements":norms,"native_cases":results}],
              "passed":all(r['passed'] for r in results)}
    write_once_json(OUT/'native-report.json',report)
    if not report['passed']:
        raise SystemExit("Native export parity failed")


if __name__ == '__main__':
    main()
