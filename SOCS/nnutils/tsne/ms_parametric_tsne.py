import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from nnutils.tsne.utils import x_to_p
from nnutils.tsne.parametric_tsne import ParametricTSNE
from torch.utils.tensorboard import SummaryWriter


class MultiscaleParametricTSNE(ParametricTSNE):

    def __init__(self,
                model: nn.Module,
                writer: SummaryWriter,
                optimizer: torch.optim.Optimizer = Adam,
                device: str = "cuda:0",
                lr: float = 1e-3,
                n_components: int = 2,
                perplexity: float = 30.0,
                n_iter: int = 100,
                batch_size: int = 500,
                early_exaggeration_epochs: int = 50,
                early_exaggeration_value: float = 4.0,
                early_stopping_epochs: float = np.inf,
                early_stopping_min_improvement: float = 1e-2,
                alpha: float = 1.0,
                verbose = 0):
        super().__init__(model,
                         writer,
                         optimizer,
                         device,
                         lr,
                         n_components,
                         perplexity,
                         n_iter,
                         batch_size,
                         early_exaggeration_epochs,
                         early_exaggeration_value,
                         early_stopping_epochs,
                         early_stopping_min_improvement,
                         alpha,
                         verbose)
    
    def _calculate_p(self, x: torch.Tensor) -> torch.Tensor:
        """
            Calculate the P values for data.

            :param x: The training data.
        """
        n = x.size(0)
        p = torch.zeros([n, self.batch_size], dtype=torch.float64)
        h = torch.round(torch.log2(torch.tensor(n/2)))
        for h_i in tqdm(torch.arange(1, h+1)):
            perplexity = 2**h_i
            for i in torch.arange(0, n, self.batch_size):
                p_batch = x_to_p(x[i:i + self.batch_size], perplexity=perplexity.item())
                p_batch[torch.isnan(p_batch)] = 0
                p_batch = p_batch + p_batch.T
                p_batch = p_batch / p_batch.sum()
                p_batch = torch.max(p_batch, torch.tensor(1e-12))
                p[i:i + self.batch_size] += p_batch
        return p/h