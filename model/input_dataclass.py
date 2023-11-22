import torch
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Any

""" We use a dataclass to define the input & state variables to keep our code concise and readable.
    The trade-off is that you have to be more cautious about stale references in the code, to avoid memory leaks. """
def image_collator(batch):
    images = torch.stack([item['image'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch])
    return ClassDC(input_img=images, labels=labels)

class BaseDC():
    def __init__(self):
        if not hasattr(self, 'x'): # Our hidden_state, for brevity.
            self.x: torch.Tensor = None
        self.instantiate("logits")
        self.instantiate("loss")
        self.instantiate("labels")
        self.instantiate("loss_fn", CrossEntropyLoss())
    def instantiate(self, variable: str, value = None):
        """ Sets attribute references, if they don't already exist. """
        if not hasattr(self, variable):
            setattr(self, variable, value)
    @property
    def hidden_state(self) -> torch.Tensor:
        """ This is literally just a pointer to the hidden_state's alias. """
        return self.x
    def calc_loss(self) -> torch.Tensor:
        """Calculates the loss using the stored loss function"""
        self.loss = self.loss_fn(self.logits, self.labels)
        return self.loss

@dataclass
class ClassDC(BaseDC):
    """A classifier dataclass, for singleton inputs"""
    input_img: torch.LongTensor = None
    labels: Optional[Any] = None
    def __post_init__(self):
        super().__init__()