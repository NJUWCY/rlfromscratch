"""Determinism probe: isolates GPU *compute* from the training loop / env.

Run it TWICE and compare the printed checksums:
    python det_probe.py
    python det_probe.py

If the forward/backward checksums differ between two runs -> it's pure GPU
compute non-determinism (some op slipping past use_deterministic_algorithms).
If they are identical -> compute is fine and the non-determinism lives in the
training loop (RNG consumption order / sampling), which we then chase there.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np

from utils.utils import set_seed, get_best_device
from utils.networks import AtariCNNEncoder, DiscreteProbabilityActor, DiscreteVFunction


def main():
    set_seed(0)
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("deterministic:", torch.are_deterministic_algorithms_enabled())

    device = get_best_device()

    obs_shape = (4, 84, 84)
    n_actions = 6  # Pong

    # Fake spaces with just the attributes the constructors touch.
    class Box:  # observation_space
        shape = obs_shape
    class Disc:  # action_space
        n = n_actions

    encoder = AtariCNNEncoder(obs_shape, 512, initialize=True)
    actor = DiscreteProbabilityActor(Box(), Disc(), encoder, initialize=True)
    critic = DiscreteVFunction(Box(), Disc(), encoder, initialize=True)
    actor.to(device); critic.to(device)

    # Fixed, deterministic input -- no env, no sampling randomness.
    x = (torch.arange(2 * 4 * 84 * 84, dtype=torch.float32, device=device)
         .reshape(2, 4, 84, 84) % 255) / 255.0

    # ---- forward ----
    logits = actor.forward(x)
    value = critic.forward(x)
    print(f"[FWD] logits_sum = {logits.double().sum().item():.12e}")
    print(f"[FWD] value_sum  = {value.double().sum().item():.12e}")

    # ---- backward ----
    loss = logits.pow(2).mean() + value.pow(2).mean()
    loss.backward()
    gnorm = 0.0
    for p in list(actor.parameters()) + list(critic.parameters()):
        if p.grad is not None:
            gnorm += p.grad.double().pow(2).sum().item()
    print(f"[BWD] grad_norm_sq = {gnorm:.12e}")


if __name__ == "__main__":
    main()
