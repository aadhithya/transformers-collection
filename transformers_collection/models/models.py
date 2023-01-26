import evaluate
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from loguru import logger
from munch import Munch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from transformers_collection.models.base import ClassificationModel, ModelMeta


class SentimentClassifier(ModelMeta):
    def __init__(self, cfg: Munch) -> None:
        super().__init__(cfg)

        logger.info(
            f"Loading pre-trained Tokenizer from huggingface: {self.cfg.base_model}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)

        (
            self.train_loader,
            self.val_loader,
            num_classes,
        ) = self._prepare_datasets()

        logger.info(
            f"Loading pre-trained model from huggingface: {self.cfg.base_model}"
        )

        model_ = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.base_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        Optim_ = getattr(torch.optim, self.cfg.optim)

        self.metric = evaluate.load("accuracy", average=None)

        self.lightning_model = ClassificationModel(
            model_,
            Optim_,
            self.cfg.lr,
            self.tokenizer,
            self.metric,
        )

        self.trainer = pl.Trainer(
            enable_checkpointing=True,
            max_epochs=self.cfg.epochs,
            log_every_n_steps=125,
            default_root_dir=self.cfg.out_dir,
            accelerator=self.cfg.device,
        )

    def _prepare_datasets(self):
        def tokenize_fn(sample):
            return self.tokenizer(sample["text"], truncation=True)

        datasets = load_dataset(self.cfg.dataset)
        train_dataset = datasets["train"]
        val_dataset = datasets["validation"]
        logger.info("Loaded train and val datasets.")

        logger.info("Tokenizing train and val datasets.")
        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        val_dataset = val_dataset.map(tokenize_fn, batched=True)
        logger.info("Tokenizing train and val datasets. Done.")

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        train_dataset = train_dataset.remove_columns("text")
        val_dataset = val_dataset.remove_columns("text")

        num_classes = train_dataset.features["labels"].num_classes
        data_collator = DataCollatorWithPadding(
            self.tokenizer, return_tensors="pt"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
        )

        val_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        return train_loader, val_loader, num_classes

    def train(self):
        logger.info("Starting training... ")

        self.trainer.fit(
            self.lightning_model, self.train_loader, self.val_loader
        )

    def test(self, dataloader=None):
        logger.info("Testing model...")
        if dataloader is None:
            logger.warning(
                "No dataloader provided for testing. Using val_loader"
            )
            dataloader = self.val_loader
        self.trainer.test(self.lightning_model, dataloader)
