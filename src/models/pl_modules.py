from typing import Any

import hydra
import pytorch_lightning as pl
import torch
<<<<<<< HEAD
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics import *
=======
import transformers as tr
from torch.optim import RAdam
>>>>>>> ad95e0f8c154fae7b5977055ce475514db35b1e0

from src.data.labels import Labels


class BasePLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(self.hparams.model)

    def forward(self, batch) -> dict:
        output_dict = self.model(batch)
        return output_dict

    def metrics(self, golds, preds, split):
        preds = torch.round(preds)
        f1 = F1Score(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        perc_ones_gold = 100 * (golds.sum() / golds.shape[0]).item()
        perc_ones_pred = 100 * (preds.sum() / preds.shape[0]).item()
        return {split + "/f1_score": f1(preds, golds),
                    split + "/precision": precision(preds, golds),
                    split + "/recall": recall(preds, golds),
                    split + "/perc_ones_gold": perc_ones_gold,
                    split + "/perc_ones_pred": perc_ones_pred,
                    }


    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch)
        self.log("train/loss", forward_output["loss"], on_step=True)
        self.log_dict(self.metrics(forward_output["gold"], forward_output["pred"], split="train"))
        return forward_output["loss"]


    def validation_step(self, batch: dict, batch_idx: int) :
        result = self.forward(batch)
        self.log("val/loss", result['loss'])
        return self.metrics(result["gold"], result["pred"], split="val"), result["loss"]

    def validation_epoch_end(self, outputs) :
        avg_val_loss = []
        avg_metrics = []
        for metrics, loss in outputs:
            avg_val_loss.append(loss)
            avg_metrics.append(metrics)

        avg_metrics = {k: sum([dic[k] for dic in avg_metrics])/len([dic[k] for dic in avg_metrics]) for k in avg_metrics[0]}
        avg_metrics["val/avg_loss"] = torch.stack(avg_val_loss).mean()
        self.log_dict(avg_metrics)

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(batch)
        self.log("test/loss", forward_output["loss"])

    def configure_optimizers(self):
<<<<<<< HEAD
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        return opt
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt),
                "interval": "step",
                "frequency": 1,
            },
        }
=======
        param_optimizer = list(self.named_parameters())
        if self.hparams.optim_params.optimizer == "radam":
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optim_params.weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = RAdam(
                optimizer_grouped_parameters, lr=self.hparams.optim_params.lr
            )
        elif self.hparams.optim_params.optimizer == "fuseadam":
            try:
                from deepspeed.ops.adam import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install DeepSpeed (`pip install deepspeed`) to use FuseAdam optimizer."
                )

            optimizer = FusedAdam(self.parameters())
        elif self.hparams.optim_params.optimizer == "deepspeedcpuadam":
            try:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
            except ImportError:
                raise ImportError(
                    "Please install DeepSpeed (`pip install deepspeed`) to use DeepSpeedCPUAdam optimizer."
                )

            optimizer = DeepSpeedCPUAdam(self.parameters())
        elif self.hparams.optim_params.optimizer == "adafactor":
            optimizer = tr.Adafactor(
                self.parameters(),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=self.hparams.optim_params.lr,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optim_params.optimizer}")

        if self.hparams.optim_params.use_scheduler:
            lr_scheduler = tr.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.optim_params.num_warmup_steps,
                num_training_steps=self.hparams.optim_params.num_training_steps,
            )
            return [optimizer], [lr_scheduler]

        return optimizer
>>>>>>> ad95e0f8c154fae7b5977055ce475514db35b1e0
