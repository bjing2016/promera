# Adapted from https://github.com/jwohlwend/boltz
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import initialize as init


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)
        multiplicity : int, optional
            The diffusion batch size, by default 1

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Caching z projection during diffusion roll-out
        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)

            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # Fused scaled dot-product attention. The pair bias z and the key
        # pad mask collapse into a single additive attention bias, and the
        # whole attention runs in the ambient (autocast) dtype via a flash /
        # memory-efficient kernel. The previous implementation pinned this
        # core to fp32 (autocast disabled + q/k/v.float()), which left the
        # dominant pairformer/diffusion attention on TF32 and added large
        # cast-copy overhead, so enabling bf16 elsewhere gave little speedup.
        # SDPA's default scale is 1/sqrt(head_dim), matching the old code.
        # q/k/v: (B, L, H, D) -> (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # z: (B, H, Lq, Lk) pair bias; mask: (B, Lk) key pad mask
        attn_bias = z + (1 - mask[:, None, None].to(z.dtype)) * -self.inf
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        # (B, H, L, D) -> (B, L, H * D)
        o = o.transpose(1, 2).reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o
