"""
PRODUCTION INFERENCE — pix2pix UHI Temperature (T) Predictor
==============================================================

This script is intended exclusively for INFERENCE. The entire pytorch-CycleGAN-and-pix2pix repository is not required in production. The generator architecture is implemented directly in this self-contained file, copied from models/networks.py.

Training command (reference, from which this script was derived):
    python train.py --dataroot ... --model pix2pix --dataset_mode uhi_npy 
        --input_nc 12 --output_nc 1 --norm_stats_json ... 
        --netG unet_512 --netD n_layers --n_layers_D 1 ...

IMPORTANT ARCHITECTURE NOTE:
    The --norm option was not explicitly specified in the training command. However,
    Pix2PixModel.modify_commandline_options() sets norm="batch" as the DEFAULT
    when --model pix2pix is selected, overriding the base default of "instance".
    Therefore, the actual trained generator uses BatchNorm2d, not InstanceNorm2d.

    If you encounter a "size mismatch" or "missing keys" error while loading
    the model, first check the checkpoints_dir/<name>/opt.txt file, if it was
    saved, to confirm the values of norm: and ngf:. Update the NORM_TYPE / NGF
    constants below accordingly.

Model input order (12 channels) — matches data/uhi_npy_dataset.py exactly:
    0-5   : Geometry one-hot encoding (6 classes, IDs 1-6 mapped to channels 0-5)
    6     : Auxiliary field (U), normalized to [-1, 1]
    7     : Valid/fluid mask ({-1, 1}) -- solid (building) = -1, fluid = 1
    8     : Target field (T) inlet value, normalized to [-1, 1] (0 where not inlet)
    9     : Target field (T) inlet mask ({-1, 1}) -- known = 1, unknown = -1
    10    : Auxiliary field (U) inlet value, normalized to [-1, 1]
    11    : Auxiliary field (U) inlet mask ({-1, 1})

Usage:
    python predict_T.py ^
        --geometry   NASH00040_10800_geometry_labels.npy ^
        --u_label    NASH00040_10800_U_labels.npy ^
        --u_inlet    NASH00040_10800_U_inlet_labels.npy ^
        --t_inlet    NASH00040_10800_T_inlet_labels.npy ^
        --checkpoint "C:\\...\\model\\latest_net_G.pth" ^
        --norm_stats "C:\\...\\model\\norm_stats.json" ^
        --out_dir    "C:\\...\\predictions" ^
        --case_name  NASH00040 --timestep 10800

Output:
    <out_dir>/<case_name>_<timestep>_T_predicted.npy
        Float32 array with NaN values at building locations.

    <out_dir>/<case_name>_<timestep>_T_predicted.png
        Temperature prediction visualization using the jet colormap.

    <out_dir>/<case_name>_<timestep>_T_predicted_meta.json
        Metadata associated with the prediction.
"""

import os
import json
import argparse
import functools

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---- must match rasterize_obj.py CLASSES / data/uhi_npy_dataset.py ----
NUM_GEOMETRY_CLASSES = 6
BUILDING_CLASS_ID = 6

# ---- architecture constants (see note above) ----
NORM_TYPE = "batch"   # "batch" or "instance" -- confirm against opt.txt if load fails
NGF = 64
NUM_DOWNS = 9         # unet_512 -> num_downs=9 (512 / 2**9 = 1x1 bottleneck)
INPUT_NC = 12
OUTPUT_NC = 1


# ============================================================
# Generator architecture (copied from models/networks.py, inference-only)
# ============================================================
def get_norm_layer(norm_type="batch"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f"norm type [{norm_type}] not supported here")


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                              norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                  norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                              outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Normalization helpers (must mirror data/uhi_npy_dataset.py exactly)
# ============================================================
def normalize(arr, lo, hi):
    """Linearly map FIXED global [lo, hi] -> [-1, 1], same formula as training."""
    return np.clip((arr - lo) / max(hi - lo, 1e-9) * 2.0 - 1.0, -1.0, 1.0)


def denormalize(arr_norm, lo, hi):
    """Inverse of normalize(): [-1, 1] -> physical units."""
    return (arr_norm + 1.0) / 2.0 * (hi - lo) + lo


def onehot_geometry(geo):
    h, w = geo.shape
    onehot = np.zeros((NUM_GEOMETRY_CLASSES, h, w), dtype=np.float32)
    for c in range(NUM_GEOMETRY_CLASSES):
        onehot[c] = (geo == (c + 1)).astype(np.float32)
    return onehot


# ============================================================
# Build the 12-channel input tensor A from the 4 available inputs
# ============================================================
def build_input_tensor(geometry_path, aux_path, target_inlet_path, aux_inlet_path,
                        stats, target_field="T", aux_field="U"):
    geo = np.load(geometry_path)                                   # (H, W) uint8, 1..6, no NaN
    aux = np.load(aux_path).astype(np.float32)                     # (H, W) float32, NaN at building

    valid_mask = np.isfinite(aux)                                  # fluid=True, solid(building)=False
    # (aux and target are always NaN-masked identically at the same building
    # pixels in field_to_2d_aligned.py, so isfinite(aux) == isfinite(target)
    # without needing the (unavailable, model-output) target array itself.)

    aux_lo, aux_hi = stats[aux_field]["min"], stats[aux_field]["max"]
    target_lo, target_hi = stats[target_field]["min"], stats[target_field]["max"]

    aux_clean = np.where(valid_mask, aux, aux_lo)
    geo_onehot = onehot_geometry(geo)                                          # (6, H, W)
    aux_norm = normalize(aux_clean, aux_lo, aux_hi)[None]                      # (1, H, W)
    mask_ch = (valid_mask.astype(np.float32) * 2.0 - 1.0)[None]               # (1, H, W)

    target_inlet = np.load(target_inlet_path).astype(np.float32)              # NaN except inlet edge(s)
    target_inlet_known = np.isfinite(target_inlet)
    target_inlet_clean = np.where(target_inlet_known, target_inlet, target_lo)
    target_inlet_norm = normalize(target_inlet_clean, target_lo, target_hi)[None]
    target_inlet_mask_ch = (target_inlet_known.astype(np.float32) * 2.0 - 1.0)[None]

    aux_inlet = np.load(aux_inlet_path).astype(np.float32)
    aux_inlet_known = np.isfinite(aux_inlet)
    aux_inlet_clean = np.where(aux_inlet_known, aux_inlet, aux_lo)
    aux_inlet_norm = normalize(aux_inlet_clean, aux_lo, aux_hi)[None]
    aux_inlet_mask_ch = (aux_inlet_known.astype(np.float32) * 2.0 - 1.0)[None]

    A = np.concatenate(
        [geo_onehot, aux_norm, mask_ch, target_inlet_norm, target_inlet_mask_ch,
         aux_inlet_norm, aux_inlet_mask_ch],
        axis=0,
    )  # (12, H, W)

    return A.astype(np.float32), valid_mask, geo


# ============================================================
# Load generator + checkpoint
# ============================================================
def load_generator(checkpoint_path, device):
    norm_layer = get_norm_layer(NORM_TYPE)
    net = UnetGenerator(INPUT_NC, OUTPUT_NC, NUM_DOWNS, ngf=NGF, norm_layer=norm_layer, use_dropout=True)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    net.load_state_dict(state_dict)   # raises a clear error if architecture doesn't match
    net.to(device)
    net.eval()
    # NOTE: forcing BatchNorm2d back into train() mode at inference (to use
    # per-sample batch stats instead of the running mean/var) was tried as an
    # experiment and measured WORSE on real data (MAE/RMSE rose 20-124% across
    # all 4 timesteps vs plain eval()) -- reverted. See conversation notes:
    # with a 9-level UNet, the deepest layers have too few spatial elements
    # (2x2, 4x4) for a single-sample batch statistic to be stable; the noisy
    # per-sample stat corrupts activations more than the pooled running
    # average helps. Do not re-apply this without re-measuring.
    return net


# ============================================================
# Post-processing: 4-stage smoothing pipeline for the prediction image
# ============================================================
def smooth_prediction_rgb(rgb_uint8, downscale_factor=4,
                           nlm_h=10, nlm_h_color=10, nlm_template=7, nlm_search=21,
                           bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                           edge_sigma_s=60, edge_sigma_r=0.4):
    """Apply a 4-stage smoothing pipeline to an RGB uint8 image (H, W, 3):

    1. Non-Local Means Denoising (cv2.fastNlMeansDenoisingColored)
       -- fine dot/speckle noise ke liye best, structure preserve karte hue
          noise hataata hai.
    2. Area-downscale -> Bicubic upscale
       -- blocky/tile artifacts (low-res model output ko crudely upsample
          karne se aane wale) todne ke liye.
    3. Bilateral filter
       -- edges/gradient boundaries preserve karte hue smooth karta hai.
    4. Edge-preserving recursive filter (cv2.edgePreservingFilter, RECURS_FILTER)
       -- final polish, gradient ko natural/smooth banata hai bina detail khoye.

    Returns an RGB uint8 image of the SAME shape as the input.
    """
    # ---- 1. Non-Local Means Denoising (color) ----
    step1 = cv2.fastNlMeansDenoisingColored(
        rgb_uint8, None, h=nlm_h, hColor=nlm_h_color,
        templateWindowSize=nlm_template, searchWindowSize=nlm_search,
    )

    # ---- 2. Area-downscale -> Bicubic upscale ----
    h, w = step1.shape[:2]
    small_w = max(1, w // downscale_factor)
    small_h = max(1, h // downscale_factor)
    downscaled = cv2.resize(step1, (small_w, small_h), interpolation=cv2.INTER_AREA)
    step2 = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    # ---- 3. Bilateral filter ----
    step3 = cv2.bilateralFilter(step2, d=bilateral_d,
                                 sigmaColor=bilateral_sigma_color,
                                 sigmaSpace=bilateral_sigma_space)

    # ---- 4. Edge-preserving recursive filter ----
    step4 = cv2.edgePreservingFilter(step3, flags=cv2.RECURS_FILTER,
                                      sigma_s=edge_sigma_s, sigma_r=edge_sigma_r)

    return step4


CLEAN_SIGMA = 8.0      # smoothing strength (px) applied to the field before saving / drawing -- raise for even smoother
COLORBAR_PAD = 0.10    # colourbar extends 10 % of (max-min) below the min and above the max


def clean_field(field, valid_mask, sigma=CLEAN_SIGMA, clip_pct=(0.5, 99.5)):
    """Remove model artefacts so that the colour range is REAL, visible data.

    1. light Gaussian smoothing on the scalar field (buildings excluded, no bleeding into the fluid)
    2. clip the extreme tails (0.5 % / 99.5 %) -- a handful of spike pixels (corners, building edges)
       would otherwise decide min/max and show up neither in the image nor anywhere you can see.
    After this, min/max of the returned field = lowest/highest values that cover a visible area, and the same
    field is saved to .npy, drawn in the PNG and reported in the summary -> colourbar == npy min..max.
    """
    f = np.where(valid_mask, field, np.nan).astype(np.float32)
    if sigma and sigma > 0:
        num = cv2.GaussianBlur(np.where(valid_mask, f, 0.0).astype(np.float32), (0, 0), sigma)
        den = cv2.GaussianBlur(valid_mask.astype(np.float32), (0, 0), sigma)
        f = np.where(valid_mask, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    fin = f[np.isfinite(f)]
    if fin.size:
        lo, hi = np.percentile(fin, clip_pct)
        f = np.where(np.isfinite(f), np.clip(f, lo, hi), np.nan).astype(np.float32)
    return f


def render_prediction_image(field_array, valid_mask, vmin, vmax, out_path,
                             field_name="T", units="K", cmap="jet",
                             smooth=True, dpi=200, title=None):
    """field_array (H, W) with NaN at solid/building pixels -> PNG with an exact-scale colorbar.

    * The colormap is applied by matplotlib directly on the VALUES (no RGB smoothing / denoising), so every
      colour in the image matches the colorbar and the colorbar always spans the field's min..max
      (vmin/vmax are taken from the field itself; the vmin/vmax arguments are only a fallback).
    * Buildings are flat dark grey (never a colormap colour), with a legend entry.
    * `smooth` is kept for compatibility and is ignored.
    """
    field = np.where(valid_mask, field_array, np.nan).astype(np.float32)
    finite = field[np.isfinite(field)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    _pad = COLORBAR_PAD * (vmax - vmin)    # colourbar: 10 % extra below min and above max
    vmin, vmax = vmin - _pad, vmax + _pad

    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("#2b2b2b")
    masked = np.ma.masked_invalid(field)

    H, W = field.shape
    fig_h = 6.5
    fig, ax = plt.subplots(figsize=(fig_h * W / H + 1.4, fig_h))
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    cbar.set_label(f"{field_name} ({units})" if units and units != "unknown" else field_name)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return masked


# ============================================================
# Main prediction routine
# ============================================================
def predict_T(geometry_path, aux_path, target_inlet_path, aux_inlet_path,
              checkpoint_path, norm_stats_path, out_dir, case_name="case",
              timestep="0", target_field="T", aux_field="U", device=None,
              smooth=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(norm_stats_path) as f:
        stats = json.load(f)

    A, valid_mask, geo = build_input_tensor(
        geometry_path, aux_path, target_inlet_path, aux_inlet_path,
        stats, target_field=target_field, aux_field=aux_field,
    )
    A_t = torch.from_numpy(A).unsqueeze(0).to(device)   # (1, 12, H, W)

    net = load_generator(checkpoint_path, device)
    with torch.no_grad():
        fake_B_norm = net(A_t)                          # (1, 1, H, W), in [-1, 1] (tanh output)

    fake_B_norm = fake_B_norm.squeeze(0).squeeze(0).cpu().numpy()   # (H, W)
    target_lo, target_hi = stats[target_field]["min"], stats[target_field]["max"]
    T_pred = denormalize(fake_B_norm, target_lo, target_hi)         # back to physical units (K)

    # Match the convention of the real field npy files: NaN at solid/building pixels.
    T_pred_masked = np.where(valid_mask, T_pred, np.nan)
    T_pred_masked = clean_field(T_pred_masked, valid_mask)   # smooth + trim spike pixels -> npy == image == colourbar

    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{case_name}_{timestep}_{target_field}_predicted"
    npy_path = os.path.join(out_dir, f"{prefix}.npy")
    png_path = os.path.join(out_dir, f"{prefix}.png")
    meta_path = os.path.join(out_dir, f"{prefix}_meta.json")

    np.save(npy_path, T_pred_masked.astype(np.float32))

    # ---- image with temperature colorbar + 4-stage smoothing pipeline ----
    finite_vals = T_pred_masked[np.isfinite(T_pred_masked)]
    vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    units = "K" if target_field == "T" else ("m/s" if target_field == "U" else "unknown")
    try:
        render_prediction_image(
            T_pred_masked, valid_mask, vmin, vmax, png_path,
            field_name=target_field, units=units, cmap="jet",
            smooth=smooth, title=f"{case_name} @ t={timestep} ({target_field})",
        )
    except Exception as e:
        print(f"[warn] PNG rendering skipped ({e})")

    meta = {
        "case": case_name,
        "timestep": timestep,
        "field": target_field,
        "units": "K" if target_field == "T" else "unknown",
        "value_min": float(vmin),
        "value_max": float(vmax),
        "model_checkpoint": os.path.abspath(checkpoint_path),
        "norm_stats": os.path.abspath(norm_stats_path),
        "note": "Predicted by pix2pix generator (unet_512, 12-channel input). "
                "NaN pixels = solid/building region.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{case_name}][{timestep}] predicted {target_field} range: {vmin:.4f} - {vmax:.4f}")
    print(f"[{case_name}][{timestep}] saved -> {npy_path}")
    return {"npy": npy_path, "png": png_path, "meta": meta_path, "array": T_pred_masked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production inference: predict T from 4 available inputs")
    parser.add_argument("--geometry", required=True, help="path to <case>_<ts>_geometry_labels.npy")
    parser.add_argument("--u_label", required=True, help="path to <case>_<ts>_U_labels.npy (aux field)")
    parser.add_argument("--t_inlet", required=True, help="path to <case>_<ts>_T_inlet_labels.npy")
    parser.add_argument("--u_inlet", required=True, help="path to <case>_<ts>_U_inlet_labels.npy")
    parser.add_argument("--checkpoint", required=True, help="path to latest_net_G.pth")
    parser.add_argument("--norm_stats", required=True, help="path to norm_stats.json")
    parser.add_argument("--out_dir", required=True, help="where to save the predicted T npy/png/meta")
    parser.add_argument("--case_name", default="case")
    parser.add_argument("--timestep", default="0")
    parser.add_argument("--target_field", default="T")
    parser.add_argument("--aux_field", default="U")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto-detect)")
    parser.add_argument("--no_smooth", action="store_true",
                         help="disable the 4-stage smoothing pipeline (NLM -> area/bicubic -> "
                              "bilateral -> edge-preserving) and save the raw colormap image instead")
    args = parser.parse_args()

    predict_T(
        geometry_path=args.geometry,
        aux_path=args.u_label,
        target_inlet_path=args.t_inlet,
        aux_inlet_path=args.u_inlet,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        out_dir=args.out_dir,
        case_name=args.case_name,
        timestep=args.timestep,
        target_field=args.target_field,
        aux_field=args.aux_field,
        device=args.device,
        smooth=not args.no_smooth,
    )