import os
import argparse
import torch
from PIL import Image
import tqdm

# Optional speedups
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_distribution(img_dir: str, apply_egovlp_preprocess: bool = True):
    """
    Compute the in-domain target distribution over all images in img_dir, as
    specified in paper Section 3 / 4.1:

        mu_Ego4D    in R^3        (per-channel mean)
        Sigma_Ego4D in R^{3x3}    (full RGB covariance, incl. cross-channel terms)

    Stats are accumulated in streaming form to avoid materializing every pixel.
    Pixels are kept in [0, 255] to match the scale used by local_cov_align /
    neighborhood_normalization.

    Returns:
        mu:  torch.Tensor, shape (3,)
        cov: torch.Tensor, shape (3, 3)
    """
    import numpy as np

    n = 0
    sum_x = np.zeros(3, dtype=np.float64)
    sum_xx = np.zeros((3, 3), dtype=np.float64)

    for fname in tqdm.tqdm(os.listdir(img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            continue
        try:
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')

            # Paper Sec 3: stats are computed after EgoVLP preprocessing
            # (resize short side -> 256, center crop -> 256, resize -> 224),
            # but BEFORE the per-channel ImageNet-style normalization, so that
            # mu/Sigma live in raw [0, 255] pixel space.
            if apply_egovlp_preprocess:
                w, h = img.size
                short = min(w, h)
                new_w = int(round(w * 256 / short))
                new_h = int(round(h * 256 / short))
                img = img.resize((new_w, new_h), Image.BILINEAR)
                left = (new_w - 256) // 2
                top = (new_h - 256) // 2
                img = img.crop((left, top, left + 256, top + 256))
                img = img.resize((224, 224), Image.BILINEAR)

            x = np.asarray(img, dtype=np.float64).reshape(-1, 3)  # N,3 in [0,255]
            n += x.shape[0]
            sum_x += x.sum(axis=0)
            sum_xx += x.T @ x
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if n == 0:
        raise RuntimeError(f"No valid images found in {img_dir}")

    mu_np = sum_x / n
    cov_np = sum_xx / n - np.outer(mu_np, mu_np)
    cov_np = 0.5 * (cov_np + cov_np.T)  # symmetrize against numerical drift

    mu = torch.from_numpy(mu_np).float()
    cov = torch.from_numpy(cov_np).float()
    return mu, cov


# -----------------------------
# Fast local mean via box filter
# -----------------------------
def box_mean(img: torch.Tensor, k: int) -> torch.Tensor:
    """
    img: H,W or H,W,C float32
    returns local mean with reflect padding, SAME shape as input.
    """
    if k <= 1:
        return img.clone()

    pad = k // 2
    
    if img.ndim == 2:
        img = img.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    
    # Reflect pad
    x = torch.nn.functional.pad(img.permute(2, 0, 1).unsqueeze(0), 
                                 (pad, pad, pad, pad), mode="reflect")  # 1,C,H+2p,W+2p
    
    # Unfold and mean
    patches = torch.nn.functional.unfold(x, kernel_size=k, padding=0)  # 1,C*k*k,num_patches
    out = patches.mean(dim=1).view(1, img.shape[2], img.shape[0], img.shape[1])
    out = out.squeeze(0).permute(1, 2, 0)
    
    if squeeze:
        out = out.squeeze(-1)
    
    return out


# -----------------------------
# Global target stats μ, Σ
# -----------------------------
def global_mean_cov(img255: torch.Tensor):
    """
    img255: H,W,3 float32 in [0,255]
    returns:
      mu: (3,)
      cov: (3,3)
    """
    x = img255.reshape(-1, 3).float()
    mu = x.mean(dim=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(len(x) - 1, 1)
    return mu, cov


def mat_sqrt_and_invsqrt_3x3(M: torch.Tensor, eps: float = 1e-8):
    """
    M: (...,3,3) symmetric PSD
    returns:
      sqrt(M), invsqrt(M)
    via batched eigen decomposition.
    """
    # Enforce symmetry
    M = 0.5 * (M + M.swapaxes(-1, -2))
    w, V = torch.linalg.eigh(M.float())  # (...,3), (...,3,3)

    w = torch.clamp(w, min=eps)
    sqrt_w = torch.sqrt(w)
    invsqrt_w = 1.0 / sqrt_w

    sqrtM = (V * sqrt_w[..., None, :]) @ V.swapaxes(-1, -2)
    invsqrtM = (V * invsqrt_w[..., None, :]) @ V.swapaxes(-1, -2)

    return sqrtM, invsqrtM


# -----------------------------
# Local covariance alignment
# y(p) = sqrt(Sigma_target) * invsqrt(Sigma_local(p)+epsI) * (x(p)-mu_local(p)) + mu_target
# -----------------------------
def local_cov_align(
    img255: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    k: int = 9,
    eps: float = 1e-3,
    shrink_lambda: float = 0.10,
    chunk_rows: int = 128,
):
    """
    img255: H,W,3 float32 in [0,255]
    mu_target: (3,)
    cov_target: (3,3)
    k: neighborhood size (odd recommended)
    eps: stability added to local covariance diag
    shrink_lambda: (1-l)*Sigma + l*diag(Sigma)
    chunk_rows: process in row chunks to limit memory
    """
    if k % 2 == 0:
        raise ValueError("k should be odd (e.g., 7, 9, 11).")

    H, W, _ = img255.shape
    device = img255.device
    # Ensure target stats live on the same device as the frame
    mu_target = mu_target.to(device=device, dtype=torch.float32)
    cov_target = cov_target.to(device=device, dtype=torch.float32)

    # Precompute sqrt of target covariance once
    sqrt_cov_t, _ = mat_sqrt_and_invsqrt_3x3(cov_target.unsqueeze(0).unsqueeze(0), eps=1e-8)
    A_t = sqrt_cov_t[0, 0]  # (3,3)

    out = torch.empty_like(img255, dtype=torch.float32)

    # Local means of channels
    mu_loc = box_mean(img255, k)  # H,W,3

    # Local second moments E[x_i x_j]
    R, G, B = img255[..., 0], img255[..., 1], img255[..., 2]
    mRR = box_mean(R * R, k)
    mGG = box_mean(G * G, k)
    mBB = box_mean(B * B, k)
    mRG = box_mean(R * G, k)
    mRB = box_mean(R * B, k)
    mGB = box_mean(G * B, k)

    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        mu_c = mu_loc[r0:r1, :, :]  # h,W,3

        # E[xx^T]
        Exx = torch.empty((r1 - r0, W, 3, 3), dtype=torch.float32, device=device)
        Exx[..., 0, 0] = mRR[r0:r1, :]
        Exx[..., 1, 1] = mGG[r0:r1, :]
        Exx[..., 2, 2] = mBB[r0:r1, :]
        Exx[..., 0, 1] = Exx[..., 1, 0] = mRG[r0:r1, :]
        Exx[..., 0, 2] = Exx[..., 2, 0] = mRB[r0:r1, :]
        Exx[..., 1, 2] = Exx[..., 2, 1] = mGB[r0:r1, :]

        # Cov = E[xx^T] - mu mu^T
        mu_outer = mu_c[..., :, None] * mu_c[..., None, :]  # h,W,3,3
        cov_loc = Exx - mu_outer  # h,W,3,3

        # Shrinkage toward diagonal
        if shrink_lambda is not None and shrink_lambda > 0:
            diag = torch.diag_embed(torch.diagonal(cov_loc, dim1=-2, dim2=-1))
            cov_loc = (1.0 - shrink_lambda) * cov_loc + shrink_lambda * diag

        # Add eps I
        I = torch.eye(3, device=device)
        cov_loc = cov_loc + eps * I

        # Invsqrt local cov
        _, invsqrt_cov_loc = mat_sqrt_and_invsqrt_3x3(cov_loc, eps=1e-8)  # h,W,3,3

        # Normalize locally: y = invsqrt_cov_loc @ (x - mu_loc)
        x_c = img255[r0:r1, :, :]  # h,W,3
        xc = (x_c - mu_c).unsqueeze(-1)  # h,W,3,1
        y = invsqrt_cov_loc @ xc  # h,W,3,1

        # Recolor to target: z = A_t @ y + mu_target
        z = (A_t.unsqueeze(0).unsqueeze(0) @ y).squeeze(-1) + mu_target  # h,W,3
        out[r0:r1, :, :] = z

    return out


def neighborhood_normalization(
    video_tensor: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    neighborhood_size: int = 9,
):
    """
    video: (B,C,T,H,W)
    """

    # Permute to (B,T,H,W,C)
    video_tensor = video_tensor.permute(0, 2, 3, 4, 1)

    # Process each frame independently
    B, T, H, W, C = video_tensor.shape
    out_video = torch.empty_like(video_tensor)
    for b in range(B):
        for t in range(T):
            frame = video_tensor[b, t]  # H,W,C
            out_frame = local_cov_align(frame, mu_target, cov_target, k=neighborhood_size)
            out_video[b, t] = out_frame

    # Permute back to (B,C,T,H,W)
    out_video = out_video.permute(0, 4, 1, 2, 3)
    return out_video

def time_normalization(
    video_tensor: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    neighborhood_size: int = 9,   # unused in the temporal-only ablation
    context_length: int = 5,
    eps: float = 1e-8,
    ):
    """
    Temporal-only ablation of paper Section 4.2/4.3: for each pixel p we compute
    per-channel temporal mean/std over a CENTERED window of radius
    tau = context_length // 2 (so |T_tau(t)| = 2*tau + 1), then apply the
    first-order version of eq (3):

        y(p, t) = (x(p, t) - mu_in(p, t)) / sigma_in(p, t) * sigma_target + mu_target

    mu_target is the paper's mu_Ego4D (shape (3,)); sigma_target is derived from
    the diagonal of cov_target (i.e. the per-channel Ego4D std). This ablation
    does not use cross-channel covariance -- that's what full
    neighborhood_normalization / neighborhood_time_normalization is for.

    video_tensor: (B, C, T, H, W)
    """
    device = video_tensor.device

    mu_target_v = mu_target.to(device=device, dtype=video_tensor.dtype).view(1, 1, 1, 1, 3)
    sigma_target = torch.sqrt(
        torch.clamp(torch.diagonal(cov_target), min=0.0)
    ).to(device=device, dtype=video_tensor.dtype).view(1, 1, 1, 1, 3)

    # (B, C, T, H, W) -> (B, T, H, W, C) for channel-last ops
    x = video_tensor.permute(0, 2, 3, 4, 1).contiguous()
    B, T, H, W, C = x.shape

    tau = max(context_length // 2, 1)
    out = torch.empty_like(x)

    # Centered window T_tau(t) = [max(0, t-tau), min(T, t+tau+1)]; we clip at
    # the video boundaries rather than leave frames unnormalized as the
    # original code did.
    for t in range(T):
        t0 = max(0, t - tau)
        t1 = min(T, t + tau + 1)
        ctx = x[:, t0:t1]                          # (B, w, H, W, C)
        ctx_mean = ctx.mean(dim=1)                 # (B, H, W, C)
        ctx_std = ctx.std(dim=1, unbiased=False)   # (B, H, W, C)
        out[:, t] = (x[:, t] - ctx_mean) / (ctx_std + eps) * sigma_target.squeeze(1) + mu_target_v.squeeze(1)

    # Permute back to (B, C, T, H, W)
    return out.permute(0, 4, 1, 2, 3)


def neighborhood_time_normalization(
    video_tensor: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    neighborhood_size: int = 9,
    context_length: int = 5,
    ):
    """
    Joint spatio-temporal neighborhood normalization (paper eq 1-3).

    This is the full method: the paper defines mu_in(p, t) and Sigma_in(p, t)
    over the product window N_k(p) x T_tau(t). We approximate it in two passes:
      1) spatial k x k cross-channel whitening/recoloring (neighborhood_normalization)
      2) per-pixel temporal rematching over a centered window of length context_length.
    Both passes use the same (mu_target, cov_target) so the output lives in the
    target's Ego4D-aligned distribution.
    """
    video_tensor = neighborhood_normalization(video_tensor, mu_target, cov_target, neighborhood_size)
    video_tensor = time_normalization(video_tensor, mu_target, cov_target, neighborhood_size, context_length)
    return video_tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_to_imagenet_normalized_video(
    video_tensor: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    method: str = "neighborhood",
    imagenet_mean=IMAGENET_MEAN,
    imagenet_std=IMAGENET_STD,
    **kwargs,
) -> torch.Tensor:
    """
    Wrapper for applying the paper's normalization to an already-ImageNet-
    normalized video tensor coming out of the EgoVLP test transform pipeline.

    The dataset loaders apply ImageNet normalization at the end of their
    transform chain, but get_distribution / local_cov_align operate in raw
    [0, 255] pixel space (paper Sec 3: target stats computed on pixel values
    after EgoVLP resize/crop, BEFORE per-channel normalization). So we:
        1) invert ImageNet normalization back to [0, 255],
        2) apply the selected method (neighborhood / time / neighborhood_time),
        3) re-apply ImageNet normalization so the model sees the same scale it
           was trained on.

    video_tensor: (B, C, T, H, W), ImageNet-normalized.
    method: one of "neighborhood", "time", "neighborhood_time".
    kwargs: forwarded to the underlying method (e.g. neighborhood_size,
            context_length).
    """
    device = video_tensor.device
    dtype = video_tensor.dtype
    im_mean = torch.tensor(imagenet_mean, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    im_std = torch.tensor(imagenet_std, device=device, dtype=dtype).view(1, 3, 1, 1, 1)

    # Invert ImageNet normalization -> [0, 255]
    pixels = (video_tensor * im_std + im_mean) * 255.0

    methods = {
        "neighborhood": neighborhood_normalization,
        "time": time_normalization,
        "neighborhood_time": neighborhood_time_normalization,
    }
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(methods)}.")
    aligned = methods[method](pixels, mu_target, cov_target, **kwargs)

    # Re-apply ImageNet normalization
    return (aligned / 255.0 - im_mean) / im_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Ego4D target stats (mu, Sigma)")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory of images to compute stats from")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to save {'mu', 'cov'} as a torch .pt file")
    args = parser.parse_args()

    mu_target, cov_target = get_distribution(args.img_dir)
    print("Target Mean:", mu_target)
    print("Target Covariance:\n", cov_target)

    if args.out is not None:
        torch.save({"mu": mu_target, "cov": cov_target}, args.out)
        print(f"Saved stats to {args.out}")