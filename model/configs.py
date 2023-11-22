from transformers import PretrainedConfig
from dataclasses import dataclass

class BaseConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_kwargs(cls, **kwargs: dict):
        """ Allows passing arbitrary params for fast prototyping and testing, though they won't be added to asdict natively. """
        cfg = cls()
        [cfg.__setattr__(k, v) for k, v in kwargs.items()]
        return cfg

@dataclass
class ClassConfig(BaseConfig):
    """ hidden_size - (default: 1024)
        num_layers - (default: 5)
        num_attn_heads - (default: 8)
        patch_size - (default: 8)
        hidden_act - (default: 'silu')
        num_classes - (default: 1000) """
    hidden_size: int = 1024
    num_layers: int = 5
    patch_size: int = 8
    num_attn_heads: int = 8
    hidden_act: str = "silu"
    num_classes: int = 1000

    def __post_init__(self):
        super().__init__()
