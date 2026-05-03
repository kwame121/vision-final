"""Run the Pix2Pix generator on one image file (CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from data.mapdata import create_pairs  # noqa: E402
from models.Unet import Unet  # noqa: E402
from viz.triptych import triptych  # noqa: E402

IMG_SIZE = 256


def _variant_use_skip(variant: str) -> bool:
    if variant == "baseline":
        return False
    if variant in ("unet", "proposed"):
        return True
    raise ValueError(f"Unknown variant: {variant}")


def load_generator(checkpoint_path: Path, device: torch.device, *, use_skip_connections: bool) -> Unet:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["generator"]
    gen = Unet(in_channels=3, out_channels=3, use_skip_connections=use_skip_connections).to(device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


def _preprocess_rgb(pil: Image.Image, device: torch.device) -> torch.Tensor:
    pil = pil.convert("RGB")
    if pil.size != (IMG_SIZE, IMG_SIZE):
        pil = TF.resize(pil, (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    t = normalize(to_tensor(pil))
    return t.unsqueeze(0).to(device, non_blocking=True)


def _tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    """t: (1,3,H,W) in [0,1]"""
    t = t.squeeze(0).cpu().clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _to_01_from_tanh(out: torch.Tensor) -> torch.Tensor:
    return torch.clamp((out + 1.0) * 0.5, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generator on one image: paired pix2pix (sat|map) or satellite-only.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image (.jpg / .png)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <input_stem>_inference.png next to input)",
    )
    parser.add_argument("--resume-from", required=True, type=Path, help="Checkpoint .pt with generator weights")
    parser.add_argument(
        "--variant",
        choices=["baseline", "unet", "proposed"],
        default="proposed",
        help="Must match how the checkpoint was trained (U-Net skips)",
    )
    parser.add_argument(
        "--format",
        choices=["paired", "satellite"],
        default="paired",
        help="paired: left=satellite, right=roadmap (pix2pix maps). satellite: whole frame is input.",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    ckpt = args.resume_from.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out_path = args.output
    if out_path is None:
        out_path = inp.parent / f"{inp.stem}_inference.png"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_skip = _variant_use_skip(args.variant)
    gen = load_generator(ckpt, device, use_skip_connections=use_skip)

    full = Image.open(inp).convert("RGB")
    if args.format == "paired":
        sat_pil, map_pil = create_pairs(full)
        x = _preprocess_rgb(sat_pil, device)
        with torch.inference_mode():
            fake = gen(x)
        fake_01 = _to_01_from_tanh(fake)
        sat_01 = _to_01_from_tanh(x)
        map_tensor = _preprocess_rgb(map_pil, device)
        map_01 = _to_01_from_tanh(map_tensor)
        sat_vis = _tensor01_to_pil(sat_01)
        pred_vis = _tensor01_to_pil(fake_01)
        map_vis = _tensor01_to_pil(map_01)
        grid_pil = triptych(sat_vis, pred_vis, map_vis, size=IMG_SIZE)
        grid_pil.save(out_path, format="PNG")
    else:
        x = _preprocess_rgb(full, device)
        with torch.inference_mode():
            fake = gen(x)
        fake_01 = _to_01_from_tanh(fake)
        sat_01 = _to_01_from_tanh(x)
        grid = torch.cat([sat_01, fake_01], dim=0)
        vutils.save_image(grid, str(out_path), nrow=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
