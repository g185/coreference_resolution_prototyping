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
        self.representation_start = FullyConnectedLayer(
            input_dim=768, hidden_size=1000, output_dim=768, dropout_prob=0.3)
        self.representation_end = FullyConnectedLayer(
            input_dim=768, hidden_size=1000, output_dim=768, dropout_prob=0.3)

        #self.encoder = DistilBertModel(DistilBertConfig())

        #for param in self.model.parameters():
        #    param.requires_grad = False

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(
            self,
            batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_as_BCE_classification(batch)

    def forward_as_MSE_regression(self, batch):
        last_hidden_state = self.model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH

        representations = self.representation(last_hidden_state)  # B X S X RH

        coref_logits = representations @ representations.permute([0, 2, 1])

        coref_logits = coref_logits.squeeze(0).flatten()
        pred = torch.sigmoid(coref_logits)  # B*S*S
        gold = batch["gold_edges"].squeeze(0).flatten()  # B*S*S
        loss_function = torch.nn.MSELoss()
        loss = loss_function(gold, pred)
        output = {"pred": pred.detach(),
                  "gold": gold.detach(),
                  "loss": loss}
        return output

    def forward_as_BCE_classification(self, batch):
        last_hidden_states = self.model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x TH
        loss = []
        preds = []
        golds = []
        for lhs, om, gold in zip(last_hidden_states, batch["offset_mapping"], batch["gold_edges"]):
            idxs_start_bpe = (om[:, 0] == 0) & (om[:, 1] != 0)
            lhs = lhs[idxs_start_bpe]
            gold = gold[idxs_start_bpe][:, idxs_start_bpe]

            #representations = self.representation_start(inputs_embeds = lhs.unsqueeze(0))["last_hidden_state"].squeeze(0)# S X RH

            coref_logits = self.representation_start(
                lhs) @ self.representation_end(lhs).T
            #coref_logits = coref_logits.fill_diagonal_(0)
            coref_logits = coref_logits.flatten()
            preds.append(torch.sigmoid(coref_logits.detach()))  # S*S
            golds.append(gold.flatten().detach())  # S*S

            loss.append(torch.nn.functional.binary_cross_entropy_with_logits(
                coref_logits, gold.flatten(), pos_weight=torch.tensor(10)))
        loss = torch.stack(loss).sum()
        output = {"pred": torch.cat(preds, 0) if len(preds) > 1 else preds[0],
                  "gold": torch.cat(golds, 0) if len(golds) > 1 else golds[0],
                  "loss": loss}
        return output

    def forward_as_fra(self, batch):
        last_hidden_state = self.model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"])["last_hidden_state"]  # B x S x H

        cartesian_matrix_idxs = torch.cartesian_prod(
            torch.arange(
                0, last_hidden_state.shape[1], device=last_hidden_state.device),
            torch.arange(
                0, last_hidden_state.shape[1], device=last_hidden_state.device)
        )

        representations = self.representation(last_hidden_state)

        x_pred, y_pred = cartesian_matrix_idxs[:,
                                               0], cartesian_matrix_idxs[:, 1]

        outputs = self.cosine_similarity(representations.squeeze(
            0)[x_pred], representations.squeeze(0)[x_pred])

        gold = batch["gold_edges"][cartesian_matrix_idxs]
        output = {
            "loss": F.binary_cross_entropy_with_logits(outputs.values.flatten().unsqueeze(-1),
                                                       batch["gold_edges"].squeeze(
                                                           0).flatten()
                                                       )
        }

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Args:
            logits (`torch.Tensor`):
                The logits of the model.
            labels (`torch.Tensor`):
                The labels of the model.

        Returns:
            obj:`torch.Tensor`: The loss of the model.
        """
        # return F.cross_entropy(
        #     logits.view(-1, self.labels.get_label_size()), labels.view(-1)
        # )
        raise NotImplementedError
