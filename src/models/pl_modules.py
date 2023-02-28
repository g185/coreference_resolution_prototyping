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

        result = {}
        f1 = F1Score(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)

        if "mentions" in preds.keys():
            mentions_pred = torch.round(preds["mentions"])  
            mentions_gold = golds["mentions"] 
            perc_ones_gold = 100 * (mentions_gold.sum() / mentions_gold.shape[0] if mentions_gold.shape[0] != 0 else torch.tensor(0)).item()
            perc_ones_pred = 100 * (mentions_pred.sum() / mentions_pred.shape[0] if mentions_pred.shape[0] != 0 else torch.tensor(0)).item()
            
            result.update({split + "/mentions_f1_score": f1(mentions_pred, mentions_gold),
                split + "/mentions_precision": precision(mentions_pred, mentions_gold),
                split + "/mentions_recall": recall(mentions_pred, mentions_gold),
                split + "/mentions_perc_ones_gold": perc_ones_gold,
                split + "/mentions_perc_ones_pred": perc_ones_pred,
                })
        if "coreferences" in preds.keys():
            coreferences_pred = torch.round(preds["coreferences"])  
            coreferences_gold = golds["coreferences"] 
            perc_ones_gold = 100 * (coreferences_gold.sum() / coreferences_gold.shape[0] if coreferences_gold.shape[0] != 0 else torch.tensor(0)).item()
            perc_ones_pred = 100 * (coreferences_pred.sum() / coreferences_pred.shape[0] if coreferences_pred.shape[0] != 0 else torch.tensor(0)).item()
            
            result.update({split + "/coreferences_f1_score": f1(coreferences_pred, coreferences_gold),
                split + "/coreferences_precision": precision(coreferences_pred, coreferences_gold),
                split + "/coreferences_recall": recall(coreferences_pred, coreferences_gold),
                split + "/coreferences_perc_ones_gold": perc_ones_gold,
                split + "/coreferences_perc_ones_pred": perc_ones_pred,
                })
            
        return result

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch)
        loss_dict = forward_output["loss_dict"]
        loss_dict = {"train/" + k: v for k,v in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True)
        self.log_dict(self.metrics(forward_output["gold_dict"], forward_output["pred_dict"], split="train"))
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        result = self.forward(batch)
        self.log_dict({"val/" + k: v for k,v in result["loss_dict"].items()})
        metrics = self.metrics(result["gold_dict"], result["pred_dict"], split="val") 
        return metrics , result["loss"]

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
