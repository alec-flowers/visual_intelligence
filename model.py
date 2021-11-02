import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.ce = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        # log step metric
        self.log('train_acc', self.train_acc(y_hat, y), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log("val_loss", loss)
        # log step metric
        self.log('val_acc', self.valid_acc(y_hat, y), on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


