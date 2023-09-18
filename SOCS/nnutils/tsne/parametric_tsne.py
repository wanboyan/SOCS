import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from nnutils.tsne.utils import x_to_p
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.base import BaseEstimator, TransformerMixin


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(self,
                model: nn.Module,
                writer=None,
                optimizer: torch.optim.Optimizer = Adam,
                device: str = "cpu",
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

        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        self.writer = writer
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.early_stopping_epochs = early_stopping_epochs
        self.early_stopping_min_improvement = early_stopping_min_improvement

        self.alpha = alpha
        self.eps = torch.tensor(10e-15, requires_grad=True).to(self.device)

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        """
            Fit the model on the data x.

            :param x: The training data.
        """
        x = torch.from_numpy(x)
        m = x.size(0) % self.batch_size
        if m > 0:
            x = x[:-m]
        n_samples, _ = x.size()
        es_patience = self.early_stopping_epochs
        es_loss = np.inf
        es_stop = False

        # pre compute p
        p = self._calculate_p(x)
        epoch = 0

        while epoch < self.n_iter and not es_stop:

            p_clone = p.clone()

            if epoch < self.early_exaggeration_epochs:
                p_clone *= self.early_exaggeration_value
            
            # Training
            total_loss = 0
            n_batches = 0
            for i in range(0, n_samples, self.batch_size):
                x_batch, p_batch = x[i:i + self.batch_size], p[i:i + self.batch_size]

                output = self.model(x_batch.to(self.device).float())
                loss = self._kl_divergence(p_batch.to(self.device), output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            # Check early-stopping condition
            if loss < es_loss and torch.abs(loss - es_loss) > self.early_stopping_min_improvement:
                es_loss = loss
                es_patience = self.early_stopping_epochs
            else:
                es_patience -= 1

            if es_patience == 0:
                es_stop = True

            epoch += 1
    
        return self


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply dimensionality reduction to x

            :param x: The data to transform.
        """

        x = torch.from_numpy(x).to(self.device).float()
        x_new = self.model(x)
        return x_new

    def fit_transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
            Fit the model with X and apply the dimensionality reduction on X.

            :param x: The data to fit the model to and transform.
        """
        self.fit(x)
        x_new = self.transform(x)
        return x_new

    def _calculate_p(self, x: torch.Tensor) -> torch.Tensor:
        """
            Calculate the P values for data.

            :param x: The training data.
        """

        n = x.size(0)
        p = torch.zeros([n, self.batch_size], dtype=torch.float64)
        for i in tqdm(torch.arange(0, n, self.batch_size)):
            p_batch = x_to_p(x[i:i + self.batch_size], perplexity=self.perplexity)
            p_batch[torch.isnan(p_batch)] = 0
            p_batch = p_batch + p_batch.T
            p_batch = p_batch / p_batch.sum()
            p_batch = torch.max(p_batch, torch.tensor(1e-12))
            p[i:i + self.batch_size] = p_batch
        return p
    
    def _kl_divergence(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
            Computes the KL divergence between the probabilities of the data
            and the model predictions.

            :param p: The data probabilities.
            :param y: The parametric t-SNE model's output.
        """
        sum_y = torch.sum(y.pow(2), dim=1)
        d = sum_y + (sum_y.reshape((-1, 1)) - 2 * torch.mm(y, y.T))
        d = 1.0 + (d / self.alpha)
        q = d.pow(-(self.alpha + 1) / 2)
        q *= (1 - torch.eye(self.batch_size, requires_grad=True).to(self.device))
        q /= q.sum()
        q = torch.maximum(q, self.eps)
        c = torch.log((p + self.eps) / (q + self.eps))
        c = (p * c).sum()
        return c