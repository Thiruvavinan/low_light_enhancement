#!/usr/bin/env python
"""
scripts/regen_log_index.py
--------------------------
Regenerates the run index in EXPERIMENT_LOG.md from outputs/eval, replacing
whatever sits between the RUN-INDEX markers.

EXPERIMENT_LOG.md is a private working document (gitignored), but its index of
"every configuration that was evaluated, and what each one showed" is the part
most likely to go stale -- an evaluation gets added and the index silently
stops being a complete list. Generating it means the numbers and the row set
come from the data; only the one-line interpretations are hand-written, and
those live in NOTES below.

    python scripts/regen_log_index.py
"""
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BEGIN = "<!-- RUN-INDEX:BEGIN -->"
END = "<!-- RUN-INDEX:END -->"

# tag -> (what it tests, what it showed, where discussed)
NOTES = {
    "input": ("identity baseline; makes no-reference scores interpretable",
              "L1 is NIQE-worse than doing nothing on LIME and DICM", "2.8, 4.4"),
    "tf_reference": ("authors' released weights, ported",
                     "PSNR 16.79 vs published 16.77 -- pipeline validated", "4.1"),
    "l1": ("paper's loss, the baseline everything is measured against",
           "the reference point", "4.4"),
    "ssim": ("does an SSIM reconstruction term help?",
             "+0.13 SSIM, -0.18 LPIPS at only +0.42 dB PSNR", "4.7"),
    "uw": ("learn the loss weights by homoscedastic uncertainty",
           "converges to 1.00 : 0.65 : 0.70; improves further", "4.3"),
    "rebalanced": ("CONTROL: those same ratios as fixed weights",
                   "matches or beats `uw` on 5 of 6 -- the method was redundant", "4.3"),
    "l1_bm3d": ("BM3D on reflectance, sigma tuned by LPIPS",
                "big full-reference gain, cross-dataset NIQE worse", "4.6"),
    "ssim_bm3d": ("same, on the SSIM model", "same sign flip, smaller", "4.6, 4.7"),
    "uw_bm3d": ("same, on the uncertainty-weighted model", "same sign flip, smallest", "4.6, 4.7"),
    "rebalanced_bm3d": ("completes the 2x4 factorial",
                        "needed before the absorption figures were meaningful", "4.7"),
    "l1_bm3d_niqe": ("re-tune sigma on NIQE instead of LPIPS",
                     "recovers most of L1's harm -- but does not generalise", "4.8"),
    "ssim_bm3d_niqe": ("same re-tune, SSIM model",
                       "still worse than no denoising", "4.8"),
    "uw_bm3d_niqe": ("same re-tune, uncertainty-weighted model",
                     "WORSE than the LPIPS sigma -- refutes the L1-only claim", "4.8"),
    "rebalanced_bm3d_niqe": ("same re-tune, fixed-weight model",
                             "also worse; 8 of 8 denoised configs lose", "4.8"),
    "ssim_decomuw": ("does the uncertainty-weighted Decom-Net help downstream?",
                     "better on all 6 metrics than the paper's Decom-Net", "4.3"),
    "ssim_lowsmooth": ("CONTROL: paper weights, smoothness / 13.1",
                       "explains it in-distribution, NOT cross-dataset", "5"),
}

rows = []
for d in sorted(Path("outputs/eval").iterdir()):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    b = s["benchmarks"]
    cross = [b[x]["niqe"] for x in ("LIME", "MEF", "DICM") if "niqe" in b.get(x, {})]
    tests, showed, where = NOTES.get(d.name, ("(undocumented)", "-", "-"))
    den = s.get("denoise", "none")
    den = "-" if den in ("none", None) else "sigma=%s g=%s" % (s.get("denoise_sigma"), s.get("denoise_gamma"))
    rows.append("| `%s` | %s | %s | %.2f / %.4f / %.4f | %.2f | %s | %s |" % (
        d.name, tests, den,
        b["LOL"]["psnr"], b["LOL"]["ssim"], b["LOL"]["lpips"],
        sum(cross) / len(cross) if cross else float("nan"), showed, where))

section = "\n".join([
    BEGIN, "",
    "## 7. Run index — every evaluated configuration",
    "",
    "Generated from `outputs/eval/*/summary.json`, so it cannot drift from the",
    "data. LOL column is PSNR / SSIM / LPIPS; cross is the mean NIQE over",
    "LIME+MEF+DICM.",
    "",
    "| tag | what it tests | denoise | LOL | cross | what it showed | section |",
    "|:---|:---|:---|:---|:---:|:---|:---:|",
    *rows,
    "",
    "Every row is one `scripts/evaluate.py --tag` invocation. Rows sharing a",
    "checkpoint differ only in inference-time denoising; see `### Protocol` in",
    "`outputs/results_table.md` for the checkpoint each one used.",
    "", END,
])

p = Path("EXPERIMENT_LOG.md")
s = p.read_text(encoding="utf-8")
if BEGIN in s and END in s:
    s = s[:s.index(BEGIN)] + section + s[s.index(END) + len(END):]
else:
    s = s.rstrip() + "\n\n---\n\n" + section + "\n"
p.write_text(s, encoding="utf-8", newline="\n")
print("run index: %d configurations" % len(rows))
