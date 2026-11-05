# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot product attention with one of the fused, memory efficient kernels."""

    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
         torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        return F.scaled_dot_product_attention(q, k, v)
