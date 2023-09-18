import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from typing import Tuple

def h_beta(d: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.

        :param d: Distance vector.
        :param beta: Constant scalar.

    """
    p = torch.exp(-d * beta)
    h = torch.log(p.sum()) + beta * torch.sum(d * p) / p.sum()
    p = p / p.sum()

    return h, p

def x_to_p(x: torch.Tensor, perplexity: float = 30.0, tol: float = 1e-5, max_tries: int = 50) -> torch.Tensor:
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.

        :param x: Data to train on.
        :param perplexity: The model perplexity.
        :param tol: Tolerance.
        :param max_tries: Maximum number of tries to evaluate whether perplexity is within tolerance.
    """

    n = x.size(0)
    log_u = torch.log(torch.tensor(perplexity))
    sum_x = torch.sum(x.pow(2), dim=1)
    d = sum_x + (sum_x - 2 * torch.mm(x, x.T))
    idxs = (1 - torch.eye(n)).type(torch.bool)
    d = d[idxs].reshape((n, -1))

    p = torch.zeros((n,n)).type(torch.float64)
    beta = torch.tensor(1.0)
    beta_min = torch.tensor(-np.inf)
    beta_max = torch.tensor(np.inf)
    for i in range(n):
        
        # Compute the Gaussian kernel and entropy for the current precision
        d_i = d[i]
        h, curr_p = h_beta(d_i, beta)

        # Evaluate whether the perplexity is within tolerance
        h_diff = h - log_u
        tries = 0
        while torch.abs(h_diff) > tol and tries < max_tries:
            # If not, increase or decrease precision
            if h_diff > 0:
                beta_min = beta
                if torch.isinf(beta_max):
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if torch.isinf(beta_min):
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

            # Recompute the values
            h, curr_p = h_beta(d_i, beta)
            h_diff = h - log_u
            tries += 1
        
        # Set final row of P
        p[i, idxs[i]] = curr_p

    return p

def plot(x: np.ndarray, y: np.ndarray, num_colours: int, title: str):
    """
        Plots data coloured by the target.

        :param x: Data to plot.
        :param y: Target classes.
        :param num_colours: Number of classes.
        :param title: Plot title.
    """
    colors = cm.rainbow(np.linspace(0, 1, num_colours))
    fig, ax = plt.subplots()
    ax.set_title(title)
    for c in range(num_colours):
        ax.scatter(x[y == c, 0], x[y == c, 1], s=8, color=colors[c], alpha=.6, label=c)
    fig.tight_layout()
    plt.legend()
    plt.show()