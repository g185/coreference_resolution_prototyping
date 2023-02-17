from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import math
from torchmetrics import *

from src.common.metrics import *

class BasePLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.mention_evaluator = MentionEvaluator()

    def forward(self, batch) -> dict:
        output_dict = self.model(batch)
        return output_dict

    def metrics(self, golds, preds, split, references = None):
        preds = torch.round(preds)
            
        f1 = F1Score(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        perc_ones_gold = 100 * (golds.sum() / golds.shape[0]).item()
        perc_ones_pred = 100 * (preds.sum() / preds.shape[0]).item()
        result = {split + "/f1_score": f1(preds, golds),
                split + "/precision": precision(preds, golds),
                split + "/recall": recall(preds, golds),
                split + "/perc_ones_gold": perc_ones_gold,
                split + "/perc_ones_pred": perc_ones_pred,
                }
        
        if references != None:
            result[split + "/f1_ment_eval"], 
            result[split+"/precision_ment_eval"], 
            result[split+"/recall_ment_eval"] = self.mention_evaluator.get_prf(golds, preds, references)
        return result

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch)
        self.log("train/loss", forward_output["loss"], on_step=True)
        self.log_dict(self.metrics(
            forward_output["gold"], forward_output["pred"], split="train"))
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        result = self.forward(batch)
        self.log("val/loss", result['loss'])
        return self.metrics(result["gold"], result["pred"], split="val"), result["loss"]

    def validation_epoch_end(self, outputs):
        avg_val_loss = []
        avg_metrics = []
        for metrics, loss in outputs:
            avg_val_loss.append(loss)
            avg_metrics.append(metrics)

        avg_metrics = {k: sum([dic[k] for dic in avg_metrics])/len([dic[k]
                                                                    for dic in avg_metrics]) for k in avg_metrics[0]}
        self.log_dict(avg_metrics)

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(batch)
        self.log("test/loss", forward_output["loss"])

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters())
        return opt
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt),
                "interval": "step",
                "frequency": 1,
            },
        }
