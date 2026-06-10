from pathlib import Path

import argparse

import json

import sys



import torch

import onnx



sys.path.insert(0, "third_party/accelerated_features")

from modules.xfeat import XFeat





class XFeatNetWrapper(torch.nn.Module):

    def __init__(self, net):

        super().__init__()

        self.net = net



    def forward(self, image):

        out = self.net(image)



        if isinstance(out, dict):

            tensors = []

            for k in sorted(out.keys()):

                if torch.is_tensor(out[k]):

                    tensors.append(out[k])

            return tuple(tensors)



        if isinstance(out, (list, tuple)):

            tensors = []

            for x in out:

                if torch.is_tensor(x):

                    tensors.append(x)

            return tuple(tensors)



        return out





def get_output_names(model, dummy):

    with torch.no_grad():

        out = model(dummy)



    if torch.is_tensor(out):

        return ["output"]



    return [f"output_{i}" for i in range(len(out))]





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--out", default="deployment/xfeat_net_static_480x640.onnx")

    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--width", type=int, default=640)

    ap.add_argument("--opset", type=int, default=17)

    args = ap.parse_args()



    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)



                                                                             

    device = torch.device("cpu")

    print("Export device:", device)



    xfeat = XFeat()

    net = xfeat.net.eval().to(device)



    model = XFeatNetWrapper(net).eval().to(device)



    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    output_names = get_output_names(model, dummy)



    print("Output names:", output_names)

    print("Input shape:", list(dummy.shape))



    with torch.no_grad():

        torch.onnx.export(

            model,

            dummy,

            str(out_path),

            input_names=["image"],

            output_names=output_names,

            opset_version=args.opset,

            do_constant_folding=True,

            export_params=True,

            dynamo=False,

            training=torch.onnx.TrainingMode.EVAL,

        )



    print("Checking ONNX model...")

    onnx_model = onnx.load(str(out_path))

    onnx.checker.check_model(onnx_model)



    meta = {

        "onnx_path": str(out_path),

        "input_name": "image",

        "input_shape": [1, 3, args.height, args.width],

        "output_names": output_names,

        "height": args.height,

        "width": args.width,

        "dynamic": False,

        "opset": args.opset,

        "export_device": "cpu",

        "note": "Static ONNX export of XFeat neural backbone. Matching and homography remain outside ONNX graph.",

    }



    meta_path = out_path.parent / "xfeat_onnx_static_export_meta.json"

    meta_path.write_text(json.dumps(meta, indent=2))



    print("Exported ONNX:", out_path)

    print("Saved metadata:", meta_path)





if __name__ == "__main__":

    main()
