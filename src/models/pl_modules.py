from typing import Any

import hydra
import pytorch_lightning as pl
import torch

from torchmetrics import *

from src.common.metrics import *
from src.models.model import *

class BasePLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.mention_evaluator = MentionEvaluator()
        self.coref_evaluator = CoNLL2012CorefEvaluator()
        self.split = ""
        
    def forward(self, batch) -> dict:
        output_dict = self.model(batch)
        return output_dict

    def metrics(self, golds, preds, split):
        
        result = {}
        f1 = F1Score(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)

        if "mentions" in preds.keys():
            mentions_pred = torch.round(preds["mentions"])  
            mentions_gold = golds["mentions"] 
            perc_ones_gold = 100 * (mentions_gold.sum() / mentions_gold.shape[1] if mentions_gold.shape[1] != 0 else torch.tensor(0)).item()
            perc_ones_pred = 100 * (mentions_pred.sum() / mentions_pred.shape[1] if mentions_pred.shape[1] != 0 else torch.tensor(0)).item()
            
            result.update({split + "/mentions_f1_score": f1(mentions_pred, mentions_gold),
                split + "/mentions_precision": precision(mentions_pred, mentions_gold),
                split + "/mentions_recall": recall(mentions_pred, mentions_gold),
                split + "/mentions_perc_ones_gold": perc_ones_gold,
                split + "/mentions_perc_ones_pred": perc_ones_pred,
                })
            
        if "coreferences_matrix_form" in preds.keys():
            coreferences_pred = torch.round(preds["coreferences_matrix_form"])  
            coreferences_gold = golds["coreferences_matrix_form"] 
            perc_ones_gold = 100 * (coreferences_gold.sum() / coreferences_gold.shape[1] if coreferences_gold.shape[1] != 0 else torch.tensor(0)).item()
            perc_ones_pred = 100 * (coreferences_pred.sum() / coreferences_pred.shape[1] if coreferences_pred.shape[1] != 0 else torch.tensor(0)).item()
            
            result.update({split + "/coreference_matrix_f1_score": f1(coreferences_pred, coreferences_gold),
                split + "/coreference_matrix_precision": precision(coreferences_pred, coreferences_gold),
                split + "/coreference_matrix_recall": recall(coreferences_pred, coreferences_gold),
                split + "/coreference_matrix_perc_ones_gold": perc_ones_gold,
                split + "/coreference_matrix_perc_ones_pred": perc_ones_pred,
                })
                
        if "coreferences" in preds.keys():
            gold = self.unpad_gold_clusters(golds["coreferences"])
            mention_to_gold_clusters = extract_mentions_to_clusters(gold)
            mention_to_predicted_clusters = extract_mentions_to_clusters(preds["coreferences"])

            precision, recall, f1 = self.coref_evaluator.get_prf(preds["coreferences"], gold, mention_to_predicted_clusters, mention_to_gold_clusters)
            result[split + "/conll2012_f1_score"] = f1
            result[split + "/conll2012_precision"] = precision
            result[split + "/conll2012_recall"] = recall
            
        return result

    def unpad_gold_clusters(self, gold_clusters):
        new_gold_clusters = []
        for batch in gold_clusters:
            new_gold_clusters = []
            for cluster in batch:
                new_cluster = []
                for span in cluster:
                    if span[0].item() != -1:
                        new_cluster.append((span[0].item(), span[1].item()))
                if len(new_cluster) != 0:
                    new_gold_clusters.append(tuple(new_cluster))
        return new_gold_clusters
            

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
        avg_metrics = []
        for metrics, loss in outputs:
            avg_metrics.append(metrics)

        avg_metrics = {k: sum([dic[k] for dic in avg_metrics])/len([dic[k]
                                                                    for dic in avg_metrics]) for k in avg_metrics[0]}
        self.log_dict(avg_metrics)

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        result = self.forward(batch)
        metrics = self.metrics(result["gold_dict"], result["pred_dict"], split="test") 
        return metrics
    
    def test_epoch_end(self, outputs):
        avg_metrics = []
        for metrics in outputs:
            avg_metrics.append(metrics)

        avg_metrics = {k: sum([dic[k] for dic in avg_metrics])/len([dic[k]
                                                                    for dic in avg_metrics]) for k in avg_metrics[0]}
        self.log_dict(avg_metrics)


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
