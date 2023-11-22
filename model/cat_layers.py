import torch
from torch import nn
from transformers.utils import logging
from transformers.activations import ACT2FN

import math
from einops import rearrange

from .input_dataclass import ClassDC
from .configs import ClassConfig
from .pos_embeddings import AxialRotaryEmbedding
#Credit for borrowed code has been sharded to its own folder to keep code clean

logger = logging.get_logger(__name__)
axial_pos = AxialRotaryEmbedding()

class AxialAttention(nn.Module):
    """ Inspired by Microsoft's RetNet, we only perform attention once for each axial direction,
        effectively "chunking" our attention operations across the different axes of our image. """
    def __init__(self, config: ClassConfig, layer: int, device="cuda"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attn_heads
        self.alpha_decay = 1 - math.tanh(math.pi * layer / config.num_layers)

        with torch.no_grad():
            self.pos = axial_pos
            self.pos.create_caches(28, 28, config.hidden_size, config.num_attn_heads, group_norm=True, padding=None) # Called seperately from pos.__init__, to dodge python compiler when not implementing this class.
            self.register_buffer("gamma", (1 - torch.pow(2.0, -5.0 - torch.arange(0, self.num_heads, device=device)))[None, :, None, None, None], persistent=False) # gamma for multi-scale res

        self.g_norm = nn.GroupNorm(self.num_heads, self.num_heads)
        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=config.bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)

    def forward(self, input_dc: ClassDC): # (b. h, w, d)
        input_dc.x = self.qkv_proj(input_dc.x) # (b, h, w, dim * 3)s
        input_dc.x = rearrange(input_dc.x, "b h w (c d) -> c b h w d", c=3) 
        input_dc.x = input_dc.x + (input_dc.input_img * self.alpha_decay) # Residual image imprinting

        q, v, k = rearrange(input_dc.x, "c b h w (n d) -> c b n h w d", n=self.num_heads)
        k = k * self.gamma # Microsoft applies gamma to only the k value.
        q, k = self.pos(q), self.pos(k) # Adds positional embeddings.

        # Of note, our einops syntax for matmul is (x y) @ (y z) = (x z)
        input_dc.x = torch.einsum("b n h x y, b n h z y -> b n h x z", q, k) # (b, h, w, n, w_2) = q @ k
        # Also of note, Microsoft uses a normalization function right here, but it degrades our model's quality significantly.
        input_dc.x = torch.einsum("b n x w y, b n h y z -> b n x w z", input_dc.x, v) # (b, h, w, n, d) = qk -> v

        input_dc.x = self.g_norm(input_dc.x) # Group norm
        input_dc.x = rearrange(input_dc.x, "b n h w d -> b h w (n d)") # Fold our heads back in as well.
        input_dc.x = self.o_proj(input_dc.x)
        return input_dc

class FFN(nn.Module):
    def __init__(self, hidden_size, hidden_act):
        super().__init__()
        inter_size = hidden_size
        self.in_proj = nn.Linear(hidden_size, inter_size, bias=False) #bias intentionally false here, for speed.
        self.act_fn = ACT2FN[hidden_act]
        self.out_proj = nn.Linear(inter_size, hidden_size, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.act_fn(self.in_proj(tensor)))

class ClassDecoder(nn.Module):
    def __init__(self, config: ClassConfig, layer):
        super().__init__()
        self.config = config
        self.norm_in = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.self_attn = AxialAttention(config, layer)
        self.norm_out = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.ffn = FFN(config.hidden_size, config.hidden_act)

    def forward(self, input_dc: ClassDC) -> ClassDC:
        x = input_dc.x # A residual state
        input_dc.x = self.norm_in(input_dc.x)
        input_dc = self.self_attn(input_dc)
        input_dc.x = input_dc.x + x

        x = input_dc.x
        input_dc.x = self.norm_out(input_dc.x)
        input_dc.x = self.ffn(input_dc.x)
        input_dc.x = input_dc.x + x
        return input_dc