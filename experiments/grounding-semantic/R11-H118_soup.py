"""R11-H118 SEED-SOUP - build weight-averaged (and alpha-interpolated) checkpoints.

Uniform average of trunk + task_head across two same-recipe draws, per the
registration's mechanics: average trunk/model.safetensors tensor-by-tensor and
dann_student.pt["task_head"]; domain_head copied from parent A (training-only);
tokenizer files copied from parent A.

Usage:
  uv run python R11-H118_soup.py --a <parentA_dir> --b <parentB_dir> \
      --out <soup_dir> [--alpha 0.5]
alpha is the weight on parent B: W = (1-alpha)*A + alpha*B (0.5 = the soup).
"""

import argparse
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    a_dir, b_dir = pathlib.Path(args.a), pathlib.Path(args.b)
    out = pathlib.Path(args.out)
    (out / "trunk").mkdir(parents=True, exist_ok=True)

    ta = load_file(a_dir / "trunk" / "model.safetensors")
    tb = load_file(b_dir / "trunk" / "model.safetensors")
    assert set(ta) == set(tb), "trunk tensor sets differ"
    al = args.alpha
    soup = {k: ((1 - al) * ta[k].float() + al * tb[k].float()).to(ta[k].dtype)
            for k in ta}
    save_file(soup, out / "trunk" / "model.safetensors")

    ha = torch.load(a_dir / "dann_student.pt", map_location="cpu", weights_only=False)
    hb = torch.load(b_dir / "dann_student.pt", map_location="cpu", weights_only=False)
    ha["task_head"] = {k: ((1 - al) * ha["task_head"][k].float()
                           + al * hb["task_head"][k].float()).to(ha["task_head"][k].dtype)
                      for k in ha["task_head"]}
    torch.save(ha, out / "dann_student.pt")

    for f in a_dir.glob("*.json"):
        shutil.copy(f, out / f.name)
    for f in (a_dir / "trunk").glob("*.json"):
        shutil.copy(f, out / "trunk" / f.name)

    print(f"soup written: {out}  alpha={al}  ({len(soup)} trunk tensors averaged)")


if __name__ == "__main__":
    main()
