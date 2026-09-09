"""
evaluation/engine.py
--------------------
Runs a trained model over a dataset and scores it. One function covers both
the paired and the unpaired case: the dataset says whether ground truth
exists (by whether samples carry a "high"), and the engine picks the metrics
that are defined for it. Nothing here branches on a model or dataset *name*,
so adding a benchmark does not touch this file.

Evaluation is always at native resolution and always one image at a time --
the benchmarks contain images of differing sizes, and resizing to a common
shape would change the very statistics NIQE/BRISQUE measure.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .denoise import BM3DReflectanceDenoiser
from .metrics import LPIPSMetric, NoReferenceMetric, psnr, ssim, ssim_gray, summarise


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Write a [1,3,H,W] or [3,H,W] float tensor as an 8-bit PNG, clamped to [0,1]."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)


@torch.no_grad()
def _forward_with_oom_fallback(
    model: torch.nn.Module, low: torch.Tensor, device: str, name: str
) -> Dict[str, torch.Tensor]:
    """
    Run one image, retrying on CPU if the GPU runs out of memory.

    Enhance-Net's multi-scale concatenation holds 3*64 channels at full input
    resolution, so a single multi-megapixel image from LIME or DICM can exceed
    a 4GB card while the rest of the benchmark fits comfortably. Downscaling
    those images instead would change the very statistics the no-reference
    metric measures, and the alternative -- evaluating everything on CPU -- is
    needlessly slow. Falling back per image keeps every benchmark at native
    resolution, and the result is bit-identical either way apart from float
    round-off. Both arms hit the same images, so this cannot skew the
    comparison.
    """
    try:
        return model({"low": low.unsqueeze(0).to(device)})
    except torch.cuda.OutOfMemoryError:
        h, w = low.shape[-2:]
        print(f"    {name}: {h}x{w} does not fit on {device}, running this image on CPU")
        torch.cuda.empty_cache()
        model.cpu()
        outputs = model({"low": low.unsqueeze(0)})
        model.to(device)
        return {k: v.cpu() for k, v in outputs.items()}


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cuda",
    save_dir: Optional[Path] = None,
    lpips_metric: Optional[LPIPSMetric] = None,
    no_reference_metric: Optional[NoReferenceMetric] = None,
    save_decomposition: bool = False,
    denoiser: Optional[BM3DReflectanceDenoiser] = None,
    progress: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """
    Parameters
    ----------
    model : a RetinexNet in eval mode; forward({"low": ...}) -> {"S", ...}
    dataset : yields {"low", "name"} and optionally "high"
    save_dir : where enhanced PNGs are written (skipped if None)
    lpips_metric / no_reference_metric : pre-built so their model weights load
        once per run rather than once per image
    save_decomposition : also write R_low, I_low and I_delta, for the
        qualitative figures that show what the decomposition actually learned
    denoiser : optional BM3D post-process on reflectance (see denoise.py).
        Applied BEFORE recombination with I_delta, as the paper specifies --
        denoising the final image instead would smooth structure that the
        illumination map legitimately introduced. Purely inference-time: the
        same trained checkpoint is used with and without it.

    Returns
    -------
    (per_image rows, mean summary)
    """
    model.eval().to(device)
    rows: List[Dict[str, object]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        name = sample["name"]

        outputs = _forward_with_oom_fallback(model, sample["low"], device, name)

        if denoiser is not None and denoiser.enabled:
            # Recompute S from the denoised reflectance rather than reusing the
            # model's S, so the denoising actually reaches the scored image.
            outputs["R_low"] = denoiser(outputs["R_low"], outputs["I_low"])
            outputs["S"] = outputs["R_low"] * outputs["I_delta"]
        enhanced = outputs["S"]

        row: Dict[str, object] = {"name": name}

        if "high" in sample:
            high = sample["high"].unsqueeze(0).to(device)
            row["psnr"] = psnr(enhanced, high)
            row["ssim"] = ssim(enhanced, high)
            row["ssim_gray"] = ssim_gray(enhanced, high)
            if lpips_metric is not None:
                row["lpips"] = lpips_metric(enhanced, high)

        if no_reference_metric is not None:
            row[no_reference_metric.backend] = no_reference_metric(enhanced)

        rows.append(row)

        if save_dir is not None:
            save_image(enhanced, Path(save_dir) / f"{name}.png")
            if save_decomposition:
                save_image(outputs["R_low"], Path(save_dir) / "decomposition" / f"{name}_R.png")
                save_image(outputs["I_low"].expand(-1, 3, -1, -1),
                           Path(save_dir) / "decomposition" / f"{name}_I.png")
                save_image(outputs["I_delta"].expand(-1, 3, -1, -1),
                           Path(save_dir) / "decomposition" / f"{name}_I_delta.png")

        if progress and (index + 1) % 10 == 0:
            print(f"    {index + 1}/{len(dataset)}", flush=True)

    return rows, summarise(rows)
