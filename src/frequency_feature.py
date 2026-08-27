import torch
import torch.nn as nn
import math

class LightweightFrequencyFeature(nn.Module):
    """
    Implements a lightweight frequency-domain feature extraction as per Feature B requirements.
    This avoids using a full neural network branch, and instead relies on 2D DCT to extract
    high-frequency coefficients and compute a video-level anomaly statistic.
    """
    def __init__(self, high_freq_threshold=0.5):
        """
        high_freq_threshold: float (0 to 1). 
        The proportion of low-frequency components to mask out.
        E.g., 0.5 means the top-left quadrant (low frequencies) in the 2D DCT matrix will be zeroed out.
        """
        super(LightweightFrequencyFeature, self).__init__()
        self.high_freq_threshold = high_freq_threshold

    def get_dct_matrix(self, N, device, dtype):
        """
        Computes the 1D DCT (Discrete Cosine Transform) Type II matrix of size N x N.
        """
        n = torch.arange(N, device=device, dtype=dtype)
        k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        
        # DCT Type II
        dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
        
        # Orthogonal normalization
        dct_mat[0, :] /= math.sqrt(2)
        dct_mat *= math.sqrt(2 / N)
        return dct_mat

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_length, c, h, w) or (batch_size, c, h, w)
        Returns: frequency anomaly score of shape (batch_size,)
        """
        original_shape = x.shape
        if len(original_shape) == 5:
            batch_size, seq_length, c, h, w = original_shape
            x = x.view(batch_size * seq_length, c, h, w)
        elif len(original_shape) == 4:
            batch_size, c, h, w = original_shape
            seq_length = 1
        else:
            raise ValueError(f"Expected 4D or 5D input, got {len(original_shape)}D")

        # 1. Convert face crop to grayscale
        if c == 3:
            # Standard RGB to Grayscale conversion
            # Y = 0.2989 R + 0.5870 G + 0.1140 B
            x = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        elif c == 1:
            x = x.squeeze(1)
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")

        # x shape: (B, H, W)
        B, H, W = x.shape
        
        # 2. Apply a 2-D Discrete Cosine Transform (DCT)
        # 2D DCT can be computed via separable 1D transforms: D_H * X * D_W^T
        D_H = self.get_dct_matrix(H, device=x.device, dtype=x.dtype)
        D_W = self.get_dct_matrix(W, device=x.device, dtype=x.dtype)
        
        dct_x = torch.matmul(D_H, x)
        dct2_x = torch.matmul(dct_x, D_W.t())
        
        # 3. Separate/measure high-frequency coefficients
        # Create a high-pass mask
        mask = torch.ones_like(dct2_x)
        
        # Mask out the top-left part (low frequencies)
        h_threshold = int(H * self.high_freq_threshold)
        w_threshold = int(W * self.high_freq_threshold)
        
        # Simple block mask: zero out the low frequency square
        mask[:, :h_threshold, :w_threshold] = 0.0
        
        high_freq_coeffs = dct2_x * mask
        
        # 4. Calculate a high-frequency energy or anomaly statistic
        # Energy = mean of absolute values
        energy = torch.mean(torch.abs(high_freq_coeffs), dim=(1, 2))
        
        # 5. Aggregate the statistic across sampled frames
        energy = energy.view(batch_size, seq_length)
        
        # Average across frames to obtain a video-level frequency score
        video_score = torch.mean(energy, dim=1)
        
        return video_score
