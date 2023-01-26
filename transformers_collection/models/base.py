import pytorch_lightning as pl
import torch
from munch import Munch
from torch import nn


class ModelMeta:
    def __init__(self, cfg: Munch) -> None:
        self.cfg = cfg
        self.tokenizer = None
        self.train_loader = self.val_loader = None
        self.model = None

    def _prepare_dataset(self):
        raise NotImplementedError(
            "Function needs to be overridden by child class!"
        )


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        Optimizer,
        lr: float = 5e-5,
        tokenizer=None,
        metric=None,
    ) -> None:
        super().__init__()

        self.model = model
        self.Optimizer = Optimizer
        self.lr = lr
        self.tokenizer = tokenizer
        self.metric = metric

    def training_step(self, batch, _):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train-loss", loss)
        return loss

    def validation_step(self, batch, _):
        outputs = self.model(**batch)
        preds = self._get_predictions(outputs)
        loss = outputs.loss
        self.log("val-loss", loss)
        self.metric.add_batch(predictions=preds, references=batch["labels"])

    def validation_epoch_end(self, outputs) -> None:
        self.__log_metrics()

    def test_step(self, batch, _):
        outputs = self.model(**batch)
        preds = self._get_predictions(outputs)
        self.metric.add_batch(predictions=preds, references=batch["labels"])

    def test_epoch_end(self, out) -> None:
        self.__log_metrics()

    def __log_metrics(self):
        metrics = self.metric.compute()
        for k, v in metrics.items():
            self.log(f"val-{k}", v)

    def _get_predictions(self, outputs):
        raise NotImplementedError("Child Class needs to implement this")

    def configure_optimizers(self):
        optimizer = self.Optimizer(self.parameters(), lr=self.lr)
        return optimizer


class ClassificationModel(BaseModel):
    def _get_predictions(self, outputs):
        return torch.argmax(outputs.logits, -1)
