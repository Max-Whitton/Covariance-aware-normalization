import os
import argparse
import torch
from PIL import Image

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
def global_mean_cov(img255: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def mat_sqrt_and_invsqrt_3x3(M: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> torch.Tensor:
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
        Exx = torch.empty((r1 - r0, W, 3, 3), dtype=torch.float32, device=DEVICE)
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
        I = torch.eye(3, device=DEVICE)
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
    neighborhood_size: int = 9,
    time_mean: torch.Tensor = torch.tensor([0.5, 0.5, 0.5]),
    time_std: torch.Tensor = torch.tensor([0.2, 0.2, 0.2]),
    context_length: int = 5,
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225)
    ):
    """
    video_tensor: (B, C, T, H, W)
    """
    device = video_tensor.device
    # Ensure time stats are on the correct device and shaped for broadcasting
    time_mean = time_mean.to(device).view(1, 1, 1, 1, 3)
    time_std = time_std.to(device).view(1, 1, 1, 1, 3)

    # 2. Permute to (B, T, H, W, C) for channel-last operations
    video_tensor = video_tensor.permute(0, 2, 3, 4, 1)
    B, T, H, W, C = video_tensor.shape

    # Create a copy to store results so we don't use normalized frames 
    # to calculate the mean of subsequent frames (sliding window)
    out_video = video_tensor.clone()

    # 3. Context Length Normalization Loop
    # We iterate up to T - context_length to ensure the window stays in bounds
    for t in range(T - context_length):
        # Slice the window: (B, context_length, H, W, C)
        context_frames = video_tensor[:, t : t + context_length]
        
        # Calculate stats across the 'temporal' dimension (dim=1)
        context_mean = context_frames.mean(dim=1, keepdim=True) # (B, 1, H, W, C)
        context_std = context_frames.std(dim=1, keepdim=True)   # (B, 1, H, W, C)

        # Apply the transformation to the current frame 't'
        # Formula: Normalized = (Current - (Target_Mean - Context_Mean)) / Context_Std * Target_Std
        # Note: We use out_video to avoid in-place corruption during the loop
        diff = time_mean - context_mean
        out_video[:, t] = (video_tensor[:, t].unsqueeze(1) - diff) / (context_std + 1e-8) * time_std

    # 4. Final Standard Normalization (ImageNet stats)
    # Permute back to (B, C, T, H, W) to apply channel-wise normalization
    out_video = out_video.permute(0, 4, 1, 2, 3)
    
    # Standard normalization: (x - mean) / std
    m = torch.tensor(norm_mean).to(device).view(1, 3, 1, 1, 1)
    s = torch.tensor(norm_std).to(device).view(1, 3, 1, 1, 1)
    out_video = (out_video - m) / s

    return out_video


def neighborhood_time_normalization(
    video_tensor: torch.Tensor,
    mu_target: torch.Tensor,
    cov_target: torch.Tensor,
    neighborhood_size: int = 9,
    time_mean: torch.Tensor = torch.tensor([0.5, 0.5, 0.5]),
    time_std: torch.Tensor = torch.tensor([0.2, 0.2, 0.2]),
    context_length: int = 5,
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225)
    ):
    """
    video_tensor: (B, C, T, H, W)
    """
    video_tensor = neighborhood_normalization(video_tensor, mu_target, cov_target, neighborhood_size)
    video_tensor = time_normalization(video_tensor, mu_target, cov_target, neighborhood_size, time_mean, time_std, context_length, norm_mean, norm_std)
    return video_tensor
    # device = video_tensor.device
    # # Ensure time stats are on the correct device and shaped for broadcasting
    # time_mean = time_mean.to(device).view(1, 1, 1, 1, 3)
    # time_std = time_std.to(device).view(1, 1, 1, 1, 3)

    # # 1. Spatial Neighborhood Normalization (Assuming this function exists)
    # # Expected output: (B, C, T, H, W)
    # video_tensor = neighborhood_normalization(video_tensor, mu_target, cov_target, neighborhood_size)

    # # 2. Permute to (B, T, H, W, C) for channel-last operations
    # video_tensor = video_tensor.permute(0, 2, 3, 4, 1)
    # B, T, H, W, C = video_tensor.shape

    # # Create a copy to store results so we don't use normalized frames 
    # # to calculate the mean of subsequent frames (sliding window)
    # out_video = video_tensor.clone()

    # # 3. Context Length Normalization Loop
    # # We iterate up to T - context_length to ensure the window stays in bounds
    # for t in range(T - context_length):
    #     # Slice the window: (B, context_length, H, W, C)
    #     context_frames = video_tensor[:, t : t + context_length]
        
    #     # Calculate stats across the 'temporal' dimension (dim=1)
    #     context_mean = context_frames.mean(dim=1, keepdim=True) # (B, 1, H, W, C)
    #     context_std = context_frames.std(dim=1, keepdim=True)   # (B, 1, H, W, C)

    #     # Apply the transformation to the current frame 't'
    #     # Formula: Normalized = (Current - (Target_Mean - Context_Mean)) / Context_Std * Target_Std
    #     # Note: We use out_video to avoid in-place corruption during the loop
    #     diff = time_mean - context_mean
    #     out_video[:, t] = (video_tensor[:, t].unsqueeze(1) - diff) / (context_std + 1e-8) * time_std

    # # 4. Final Standard Normalization (ImageNet stats)
    # # Permute back to (B, C, T, H, W) to apply channel-wise normalization
    # out_video = out_video.permute(0, 4, 1, 2, 3)
    
    # # Standard normalization: (x - mean) / std
    # m = torch.tensor(norm_mean).to(device).view(1, 3, 1, 1, 1)
    # s = torch.tensor(norm_std).to(device).view(1, 3, 1, 1, 1)
    # out_video = (out_video - m) / s
