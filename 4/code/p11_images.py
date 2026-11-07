"""Phase 11 inputs: 25 synthetic object images and their PhysCtrl configs (pre-registered Oct 20, 2026; PLAN.md "Phase 11").

FLUX.1-schnell (open licence), 4 steps, 1152 x 768 then resized to 720 x 480 (PhysCtrl's example size; DAS resizes
to 480 x 720), guidance 0, seed = object index. Physics per image by the
pre-registered rule: elastic, log10 E = 4 + (i mod 3), nu = 0.3, push at the highest point (drag_mode max, axis y),
direction by i mod 3 (up, up-left, up-right). The force coefficient is set per case by p6_video.py (force_coeffs).
No visual curation: every generated image is kept; failures of PhysCtrl's own preprocessing are recorded there.
Output: $BIG_ROOT/p11_examples/<slug>/{input.png, config.json}
  --dry-run: configs only, with a flat grey stand-in image (checks the rule; no model needed).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OBJECTS = ["rubber duck", "plush teddy bear", "soft rubber toy dinosaur", "foam sponge", "jelly dessert", "stuffed bunny",
           "rubber chicken toy", "silicone spatula", "plush octopus", "soft toy car", "rubber boot", "foam letter A",
           "squishy star-shaped stress ball", "rubber frog toy", "plush dog", "stuffed elephant", "giant gummy bear",
           "plush cactus", "rubber giraffe teether", "plush fish", "foam pillow", "rubber snake toy", "plush mushroom",
           "stuffed cat", "rubber dog chew toy"]
PROMPT = "a photo of a {o} standing on a wooden table, centered, whole object visible, plain light background, studio lighting"
DIRS = [([0, 1, 0], "upwards"), ([-1, 0.3, 0], "up and to the left"), ([1, 0.3, 0], "up and to the right")]


def config(i, obj):
    d, phrase = DIRS[i % 3]
    return {"material": "elastic", "log_E": 4 + (i % 3), "nu": 0.3, "force_coeff": None, "drag_mode": "max",
            "drag_axis": 1, "drag_dir": d,
            "prompt": f"the {obj} is pushed {phrase} by an invisible force and moves naturally"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.expandvars("${BIG_ROOT}/p11_examples"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(OBJECTS) != 25 or len(set(OBJECTS)) != 25:
        raise RuntimeError("the pre-registered list has 25 distinct objects")
    out = Path(args.out)
    pipe = None
    if not args.dry_run:
        import torch
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    for i, obj in enumerate(OBJECTS):
        slug = f"p11_{i:02d}_" + obj.replace(" ", "_").replace("-", "_")
        d = out / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config(i, obj), indent=1))
        if (d / "input.png").exists():
            continue
        if args.dry_run:
            from PIL import Image
            Image.new("RGB", (64, 64), (200, 200, 200)).save(d / "input.png")
            continue
        import torch
        img = pipe(PROMPT.format(o=obj), num_inference_steps=4, guidance_scale=0.0, height=768, width=1152,
                   max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(i)).images[0]
        img.resize((720, 480), resample=3).save(d / "input.png")      # 3 = LANCZOS
        print(f"[p11 images] {slug}", flush=True)


if __name__ == "__main__":
    main()
