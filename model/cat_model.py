from transformers import PreTrainedModel
from transformers.utils import logging
from .cat_layers import *

logger = logging.get_logger(__name__)

class CatPreTrainedModel(PreTrainedModel):
    config_class = ClassConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_device(self):
        return next(self.parameters()).device

class CatHead(PreTrainedModel):
    """ Multi-channel, cross-attention, rention-based computer-vision auto-encoder. """
    def __init__(self, config: ClassConfig):
        super().__init__(config)
        self.embed_image = nn.Conv2d(3, config.hidden_size, config.patch_size, config.patch_size)
        self.layers = nn.ModuleList([ClassDecoder(config, i) for i in range(config.num_layers)])
        self.out_norm = nn.LayerNorm(config.hidden_size, bias=None)
        self.gradient_checkpointing=False
        self.post_init()
    
    def forward(self, input_dc: ClassDC):
        """ Expects tensor of size (batch_size, channels, height, width) """
        input_dc.x = rearrange(self.embed_image(input_dc.input_img), "b d h w -> b h w d")
        input_dc.input_img = input_dc.x[None, :, :, :, :].contiguous() # Explicitly removes einops' view
        for fn in self.layers:
            input_dc = fn(input_dc)
        input_dc.x = self.out_norm(input_dc.x)
        return input_dc

class CatForImageClassification(PreTrainedModel):
    """ Frames are something HF cares about. It wants explicit access to stuff like the embedding layers. """
    def __init__(self, config: ClassConfig):
        super().__init__(config)
        self.config = config # Our trainer wants this
        self.cv_head = CatHead(config)
        self.logits = nn.Linear(config.hidden_size, config.num_classes)
    
    def pool_outputs(self, tensor):
        """ We 'pool' here by just taking the corner token in the sequence. """
        return tensor[:, -1, -1, :]

    def get_input_embeddings(self):
        return self.cv_head.embed_image

    def forward(self, input_dc: ClassDC):
        input_dc = self.cv_head(input_dc)
        input_dc.logits = self.logits(self.pool_outputs(input_dc.x))
        return input_dc