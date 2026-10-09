"""Optimizers shared by the algorithms in this library."""
import math

import torch
from torch.optim import Optimizer


class OAdam(Optimizer):
    """Optimistic Adam, adapted from outputs/garage/garage/utils/oadam.py."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps and weight_decay must be nonnegative")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, amsgrad=amsgrad))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("OAdam does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(parameter)
                    state['exp_avg_sq'] = torch.zeros_like(parameter)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(parameter)
                state['step'] += 1
                if group['weight_decay']:
                    grad = grad.add(parameter, alpha=group['weight_decay'])
                first, second = state['exp_avg'], state['exp_avg_sq']
                step_size = (group['lr'] * math.sqrt(1 - beta2 ** state['step'])
                             / (1 - beta1 ** state['step']))
                # Undo the previous prediction before applying twice the new one.
                parameter.addcdiv_(first, second.sqrt().add(group['eps']), value=step_size)
                first.mul_(beta1).add_(grad, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    maximum = state['max_exp_avg_sq']
                    torch.maximum(maximum, second, out=maximum)
                    denominator = maximum.sqrt().add_(group['eps'])
                else:
                    denominator = second.sqrt().add_(group['eps'])
                parameter.addcdiv_(first, denominator, value=-2 * step_size)
        return loss
