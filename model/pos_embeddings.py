import torch
import torch.nn as nn
from einops import rearrange
from typing import Literal
from math import pi

class BaseRotaryEmbedding(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        with torch.no_grad(): # Tiny scalar to ensure model has parameters to move around, without consuming resources on compile.
            self.register_buffer("half_dim", torch.tensor(1, device=device), persistent=False)

    def negate_half(self, x):
        """Negates half the hidden dims of the input."""
        x1 = x[..., x.shape[-1] // 2 :]
        x2 = x[..., : x.shape[-1] // 2]
        return torch.cat((-x2, x1), dim=-1)
    def _apply_pos(self, q, k):
        """ The old non-cached function, preserved. """
        return (q * self.cos) + (self.negate_half(q) * self.sin), (k * self.cos) + (self.negate_half(k) * self.sin)
    
    def apply_pos(self, tensor: torch.Tensor) -> torch.Tensor:
        """ The apply function for the fully cached variant. """
        return (tensor * self.cos) + (tensor * self.sin)

class AxialRotaryEmbedding(BaseRotaryEmbedding):
    def __init__(self, breadth=1, theta_rescale=1., method: Literal["interleave", "mean"] = "interleave", device=None):
        """ Args:
            - breadth = How many radians to have spanning the largest height & width dims. 
            - theta_rescale = How fast the embeddings rotate.
            - method = How to mix the different rotation channels. Interleave is less lossy. """
        super().__init__(device)
        # We want this to be part of the initialized non-persistent state so that it gets moved w/ model weights during ddp.
        self.theta = 10000
        self.rescale = theta_rescale #Rescale factor
        self.breadth = breadth
        self.method = method

    def create_caches(self, h, w, dim, num_heads = 1, padding: Literal["left", "bottom", None] = None, group_norm: bool = False, add_channel: bool = False, device=None):
        """ Args: 
            - h: Height
            - w: Width
            - dim: Total hidden_size.
            - num_heads: If you want to cache splitting the heads, too.
            - padding: Adds a row/col to the cache for CLS/misc tokens
            - group_norm: Whether to put the num_heads dim in front of the batch dim or not.
            - add_channel: Whether to include broadcasting for a channel dimension. """
        with torch.no_grad():
            self.h, self.w = h, w

            # Borrowing the scaling factor from llama's sequence length scaling, since it seems like an interesting and non-invasive hyperparameter to experiment with.
            if self.theta == 10000 and self.rescale != 1.:
                self.theta *= self.rescale ** (dim / (dim - 2)) # Would equal 1 if rescale is 1.

            # -- Buffer for forming axes
            freq_dim = dim // 2 if self.method == "interleave" else dim #Interleave method generates half the dims itself, so our spacing accounts for this broadcast operaiton.
            freq = 1. / (self.theta ** (torch.arange(0, freq_dim, 2).float().to(device) / dim//2))
            self.register_buffer("half_dim", torch.cat((freq, freq), dim=-1), persistent=False)
            # --------

            if self.h > self.w: # Scales positions to maintain aspect ratio.
                ratio = w/h
                h = torch.linspace(self.breadth, -self.breadth, h) * pi
                w = torch.linspace(-self.breadth*ratio, self.breadth*ratio, w) * pi
            elif self.w > self.h:
                ratio = h/w
                h = torch.linspace(self.breadth*ratio, -self.breadth*ratio, h) * pi
                w = torch.linspace(-self.breadth, self.breadth, w) * pi
            else: # If identical lengths:
                h = torch.linspace(self.breadth, -self.breadth, h) * pi
                w = torch.linspace(-self.breadth, self.breadth, w) * pi
            
            h = torch.einsum("y, z -> yz", h, self.half_dim) # Now we have 'four' freq dims, which will become two when we interleave.
            w = torch.einsum("x, z -> xz", w, self.half_dim)

            """ Now we need to do some cute math to keep this stuff differentiable because loops are inefficient.
                There are two methods, interleaving and averaging. We can't be lazy and just concat, because of
                negate_half's implementation. We'd brick the rotation matrix. """

            if self.method == "interleave": # Creates a dummy dimension of zeros, stacks them, and then flattens them, so that we can sum the tensors to interleave them.
                h = torch.stack((torch.zeros((self.h, dim//2), device=device), h), dim = 2).flatten(1) # Has been unit-tested.
                w = torch.stack((w, torch.zeros((self.w, dim//2), device=device)), dim = 2).flatten(1)
            hw = h[:, None, :] + w[None, :, :]
            if self.method == "mean": # This just averages them, which loses some information, and is less faithful to the OG sinusoidal implementation.
                hw = hw/2
            cos = hw.cos()
            sin = hw.sin()
            if padding == "left": # (h, w, d), to:do update this to "row" and "col" for brevity.
                cos = torch.cat((torch.ones((self.h, 1, cos.size(-1))), cos), dim=1)
                sin = torch.cat((torch.ones((self.h, 1, sin.size(-1))), sin), dim=1)
            elif padding == "bottom":
                cos = torch.cat(cos, torch.ones((1, self.w, cos.size(-1))), dim=0)
                sin = torch.cat(sin, torch.ones((1, self.w, sin.size(-1))), dim=0)
            if num_heads > 1:
                if group_norm is False:
                    cos = rearrange(cos, "... (n d) -> ... n d", n=num_heads) # (h, w, n, d)
                    sin = rearrange(sin, "... (n d) -> ... n d", n=num_heads) # (h, w, n, d)
                elif group_norm is True:
                    cos = rearrange(cos, "... (n d) -> n ... d", n=num_heads) # (h, w, n, d)
                    sin = rearrange(sin, "... (n d) -> n ... d", n=num_heads) # (h, w, n, d)
            if add_channel:
                cos.unsqueeze(1)
                sin.unsqueeze(1)

            self.register_buffer("cos", cos.unsqueeze(0)) # adds batch_dim
            self.register_buffer("sin", self.negate_half(sin.unsqueeze(0)))
            # All operations are now FULLY CACHED. Temporal return tests at ~10%.
    
    def update_forward(self, q, k, num_heads = 1, padding: Literal["left", "bottom", None] = None, add_channel: bool = False):
        """ A forward method that checks for dynamic h & w + automatic rescales, not recommended due to excess conditionals. 
            Consider sub-classing and hard-coding for your use-case if your resolution varies. """
        if not add_channel:
            _, h, w, d = q.shape
        else:
            _, _, h, w, d = q.shape
        if padding == "left":
            w -= 1
        if padding == "right":
            h -= 1
        if h != self.h or w != self.w:
            self.create_caches(h, w, d, num_heads, padding, add_channel, device=q.device)
        return self.apply_pos(q, k)

    def forward(self, tensor):
        return self.apply_pos(tensor)