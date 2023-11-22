import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from datasets import load_dataset
import shutil

from .model.input_dataclass import image_collator

from .model.configs import ClassConfig
from .model.cat_model import CatForImageClassification
from .trainer.custom_datasets import BasicImageDataset
from .trainer.lightning_module import LightningModel

# This trainer is based on our 'Lightning ReLoRA' trainer.

# --------------------
# Project Parameters
# --------------------
project_name = "image_classification"
train_precision = "bf16-mixed" #["32", "bf16-mixed", "16-mixed", "int8"]
dev_mode = False
use_wandb = True
use_lora = False
dataset_shards = 100

# -- trainer params --
batch_size = 64
max_epochs = 10
lora_merge_epochs = 1
learning_rate = 3e-4
max_steps = None
max_val_steps = None
# --------------------

# Library variables
L.seed_everything(234)
torch.set_float32_matmul_precision("medium")

# Helper functions
def init_model(model_class):
    config = ClassConfig()
    model = model_class(config)
    model.init_weights()
    return model

def cleanup(model, mode: ["begin", "end"] = "begin"):
    if mode == "end":
        if use_lora is True:
            model.save_pretrained("final_version_lora")
            model = model.merge_and_unload()
        model.save_pretrained("final_version")
        [shutil.rmtree(f"./{dir}") for dir in [project_name, "wandb"]]

# Train
def main():
    # Dataset preperations
    train_set = BasicImageDataset(load_dataset(path="imagenet-1k", split="train"), length=1281167, shards=dataset_shards)
    val_set = BasicImageDataset(load_dataset(path="imagenet-1k", split="validation"), length=50000, shards=dataset_shards//10)

    # Model init
    model_class = init_model(CatForImageClassification)
    model_path = None

    # ------------------------------
    # Auto initialize trainer params
    # ------------------------------
    precision = train_precision
    logger = WandbLogger(project=project_name, dir=project_name) if use_wandb is True and dev_mode is False else None
    train_steps = max_steps if dev_mode is False else 10
    val_steps = max_val_steps if dev_mode is False else 5
    epochs = max_epochs if dev_mode is False else 5
    log_steps = 50 if dev_mode is False else 2
    batches = batch_size if dev_mode is False else 2


    # --- Run ---
    model = LightningModel(model_class=model_class,
                            model_path=model_path,
                            collator=image_collator,
                            batch_size=batches,
                            train_dataset=train_set,
                            eval_dataset=val_set,
                            learning_rate=learning_rate)
    trainer = L.Trainer(max_epochs=epochs,
                        precision=precision,
                        logger=logger,
                        val_check_interval=1.0,
                        log_every_n_steps=log_steps, 
                        limit_train_batches=train_steps,
                        limit_val_batches=val_steps,
                        reload_dataloaders_every_n_epochs=1,
                        default_root_dir=project_name,
                        callbacks=[LearningRateMonitor("step")])
    trainer.fit(model)
    cleanup(model.model, "end") # Unwrap model

if __name__ == "__main__":
    main()