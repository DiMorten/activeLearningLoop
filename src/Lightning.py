import pytorch_lightning as pl
import segmentation_models_pytorch_dropout as smpd
import torch
from torch.utils.data import DataLoader
# define the LightningModule
class LitModel(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # fill arguments
        parser.add_argument("--encoder_weights", type=str, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        
        self.model = smpd.DeepLabV3Plus('resnet34', encoder_weights=self.hparams['encoder_weights'], 
                in_channels=3, classes=2)


    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        y = torch.nn.functional.softmax(self.model(x))

        # Use metrics module to calculate uncertainty metric

        uncertainty_map = get_uncertainty(y)
        # Logging to TensorBoard by default
        # self.log("train_loss", loss)
        
        return {'softmax': y, 'uncertainty_map': uncertainty_map, 'uncertainty': uncertainty}

class HilaiDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # fill arguments
        parser.add_argument("--encoder_weights", type=str, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.data_dir = data_dir
        self.batch_size = batch_size
    def setup(self, stage: str):
        self.dataset_val = HilaiDataset(**cfg)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        # salvar mascara de softmax e mapa de incerteza
        # pl_module
        # outputs

        # batch: passar

from argparse import ArgumentParser
parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument("--conda_env", type=str, default="some_name")
parser.add_argument("--notification_email", type=str, default="will@email.com")
# add model specific args
parser = LitModel.add_model_specific_args(parser)
parser = HilaiDataModule.add_model_specific_args(parser)
# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)
args = vars(parser.parse_args())

# init the model
model = LitModel(**args)

dm = HilaiDataModule(**args)
dm.setup("validation")
val_dataloader = dm.val_dataloader()
trainer = pl.Trainer.from_argparse_args(args)
trainer.validate(model=model, dataloaders=val_dataloader)
 