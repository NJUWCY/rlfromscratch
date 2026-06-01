import torch
from torch.distributions import Normal,Transform,TransformedDistribution,TanhTransform,AffineTransform,Independent
import numpy as np
def correct_log_prob_gaussian_tanh(
    log_prob: torch.Tensor,
    tanh_squashed_action: torch.Tensor,
    eps: float = np.finfo(np.float32).eps.item(),
) -> torch.Tensor:
    """Apply correction for Tanh squashing when computing `log_prob` from Gaussian.

    See equation 21 in the original `SAC paper <https://arxiv.org/abs/1801.01290>`_.

    :param log_prob: log probability of the action
    :param tanh_squashed_action: action squashed to values in (-1, 1) range by tanh
    :param eps: epsilon for numerical stability
    """
    log_prob_correction = torch.log(1 - tanh_squashed_action.pow(2) + eps).sum(-1, keepdim=True)
    return log_prob - log_prob_correction


loc,scale = torch.tensor(15,dtype=torch.float32),torch.tensor(2,dtype=torch.float32)
dist = Normal(loc=loc, scale=scale)

x = dist.sample()
log_prob = dist.log_prob(x)
squashed_action = torch.tensor(1.)
log_prob = correct_log_prob_gaussian_tanh(log_prob, squashed_action)
print(x,squashed_action)
print(log_prob)

dist = Normal(loc=loc, scale=scale)
dist = TransformedDistribution(dist, [TanhTransform(cache_size=1), AffineTransform(loc=0, scale=1)])
t = dist.rsample()
print(t,dist.log_prob(t))