import torch
from torch import nn
import numpy as np
from src.common.latent.kuma_gate import KumaGate
from src.common.latent.bernoulli_gate import BernoulliGate

class IndependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self, hidden_size, sparsity):

        super(IndependentLatentModel, self).__init__()

        self.z_layer = KumaGate(768)
        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)
        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x):

        h = x
        z_dist = self.z_layer(h)
        
        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        self.z = z  # [B, T]
        self.z_dists = [z_dist]


        return z, 0

    def evaluate_latent_loss(self):
        a=1

    


