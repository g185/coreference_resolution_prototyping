from typing import Dict, List, Optional
from transformers import AutoModel, AutoConfig, DistilBertModel, DistilBertConfig
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers.activations import ACT2FN
import math


class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense1 = Linear(self.input_dim, hidden_size)
        self.dense = Linear(hidden_size, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim)
        self.activation_func = torch.nn.ReLU()
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense1(temp)
        temp = self.dropout(temp)
        temp = self.activation_func(temp)
        temp = self.dense(temp)
        return temp


class CorefModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hf_model_name = kwargs["huggingface_model_name"]
        self.model = AutoModel.from_pretrained(self.hf_model_name)
        self.config = AutoConfig.from_pretrained(self.hf_model_name)
        self.linear = kwargs["linear_layer_hidden_size"]
        self.representation_start = FullyConnectedLayer(
            input_dim=768, hidden_size=self.linear, output_dim=768, dropout_prob=0.3)
        self.representation_end = FullyConnectedLayer(
            input_dim=768, hidden_size=self.linear, output_dim=768, dropout_prob=0.3)
        self.mode = kwargs["mode"]
        self.pos_weight = kwargs["pos_weight"]
        if kwargs["transformer_freeze"] == "freezed":
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(
            self,
            batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.mode == "s2s":
            return self.forward_as_BCE_classification_s2s(batch)
        elif self.mode == "s2e":
            return self.forward_as_BCE_classification_s2e(batch)
        elif self.mode == "s2e_sentence_level":
            return self.forward_as_BCE_classification_s2e_sentence_level(batch)



    def forward_as_BCE_classification_s2s(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []

        for lhs, mask, gold in zip(last_hidden_states, batch["mask"], batch["gold_edges"]):
            lhs = lhs[mask == 1]
            gold = gold[mask == 1][:, mask == 1]

            coref_logits = self.representation_start(
                lhs) @ self.representation_end(lhs).T

            coref_logits = coref_logits

            pred = torch.sigmoid(coref_logits.flatten().detach())
            gold = gold.flatten().detach()

            preds.append(pred)  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits.flatten(), gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()
        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "loss": loss}
        return output

    def forward_as_BCE_classification_s2e_sentence_level(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        for  lhs, ids, mask, gold in zip(last_hidden_states, batch["input_ids"], batch["mask"], batch["gold_edges"]):

            eoi = (ids == 2).nonzero(as_tuple=False)
            lhs = lhs[:eoi]
            gold = gold[:eoi, :eoi]

            
            mask = mask[:eoi, :eoi]
            coref_logits = self.representation_start(
                lhs) @ self.representation_end(lhs).T
            coref_logits = coref_logits[mask==1]
            gold = gold[mask==1]
            
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()

        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "loss": loss}
        return output


    def forward_as_BCE_classification_s2e(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        for  lhs, ids, gold in zip(last_hidden_states, batch["input_ids"],  batch["gold_edges"]):

            eoi = (ids == 2).nonzero(as_tuple=False)
            lhs = lhs[:eoi]
            gold = gold[:eoi, :eoi]

            coref_logits = self.representation_start(
                lhs) @ self.representation_end(lhs).T
            
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold.flatten(), pos_weight=torch.tensor(self.pos_weight)))
        loss = torch.stack(loss).sum()

        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "loss": loss}
        return output

