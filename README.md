# RLfromScratch

A deep reinforcement learning library implemented from scratch, built on PyTorch and Gymnasium, supporting Atari and MuJoCo environments. This project aims to provide clear, modular RL algorithm implementations for learning and research purposes.

## Project Structure

```
RLfromscratch/
├── algorithm/              # Algorithm implementations
│   ├── basealgorithm.py    # Abstract base class for all algorithms
│   ├── baseoffpolicy.py    # Off-Policy algorithm base class
│   ├── baseonpolicy.py     # On-Policy algorithm base class
│   ├── dqn.py              # DQN and variants
│   ├── sac.py              # Soft Actor-Critic
│   ├── td3.py              # Twin Delayed DDPG
│   ├── ppo.py              # Proximal Policy Optimization
│   ├── trpo.py             # Trust Region Policy Optimization
│   └── bc.py               # Behavior Cloning (offline learning)
├── agent/                  # Agents (network assembly & action selection)
│   └── agent.py            # Agent implementations (DQN, A2C, SAC, TD3, etc.)
├── memory/                 # Experience replay
│   ├── memory.py           # ReplayBuffer / TrajectoryRollout
│   ├── datastructure.py    # SumTree (prioritized experience replay)
│   └── expert_dataset.py   # Expert dataset (offline RL)
├── env/                    # Environment wrappers
│   ├── make_envs.py        # Unified environment creation entry
│   ├── atari_envs.py       # Atari environment wrapper
│   ├── mujoco_envs.py      # MuJoCo environment wrapper
│   ├── basic_envs.py       # Basic environment wrapper
│   └── env_types.py        # Environment type registry
├── utils/                  # Utilities
│   ├── networks.py         # Neural networks (MLP, CNN, Actor, Critic)
│   ├── evaluate.py         # Evaluation functions
│   └── utils.py            # Helper functions (conjugate gradient, RunningMeanStd, etc.)
├── logger/                 # Logging system
│   └── logger.py           # TensorBoard / WandB logging
├── config/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── algorithm/          # Algorithm hyperparameters (dqn.yaml, ppo.yaml, ...)
│   └── env/                # Environment configs (atari.yaml, mujoco.yaml)
├── scripts/                # Experiment scripts
├── run_dqn.py              # DQN training entry
├── run_ppo.py              # PPO (MuJoCo) training entry
├── run_ppo_atari.py        # PPO (Atari) training entry
├── run_sac.py              # SAC training entry
├── run_td3.py              # TD3 training entry
├── run_trpo.py             # TRPO training entry
└── run_bc.py               # Behavior Cloning training entry
```

## Implemented Algorithms

### Off-Policy Algorithms

| Algorithm | Paper | Description |
| --- | --- | --- |
| DQN | [Mnih et al. 2015](https://www.nature.com/articles/nature14236) | Supports Double DQN, Dueling DQN, Prioritized Replay, N-step Returns |
| SAC | [Haarnoja et al. 2018](https://arxiv.org/abs/1812.05905) | Automatic temperature coefficient tuning |
| TD3 | [Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477) | Delayed policy update, target policy smoothing |

### On-Policy Algorithms

| Algorithm | Paper | Description |
| --- | --- | --- |
| PPO | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) | Clipped objective, GAE, supports both Atari and MuJoCo |
| TRPO | [Schulman et al. 2015](https://arxiv.org/abs/1502.05477) | Conjugate gradient, line search |

## Supported Environments

### Atari

Uses ALE (Arcade Learning Environment) + Gymnasium wrappers with default settings:
- Frame Stack: 4
- Frame Skip: 4
- Max Episode Length: 10000

Supported games include: Pong, Breakout, SpaceInvaders, Qbert, Seaquest, etc.

### MuJoCo

Uses Gymnasium MuJoCo environments with default settings:
- Max Episode Length: 1000
- Vectorized parallel environments (SB3 SubprocVecEnv)
- Observation Normalization support

Supported environments include: HalfCheetah, Hopper, Walker2d, Ant, Humanoid, etc.

## Benchmark

### MuJoCo Performance Comparison

The following tables compare performance across mainstream RL libraries on MuJoCo continuous control tasks.

> **RLfromScratch evaluation metric:** At each evaluation step, returns are averaged across the training seeds. Within the stated interaction budget, the five highest points on this mean curve are selected. Each seed is averaged over those same five steps, and the table reports the mean ± population standard deviation (`ddof=0`) across the resulting seed averages. All results use 10 seeds, except PPO Hopper-v5 and Humanoid-v5, which exclude the anomalous seed 5 and use 9 seeds.

#### SAC (1M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | 11231.93 ± 322.98 | 12138.8 ± 1049.3 | 9535.451 ± 100.470 | ~11520 | 9634.89 ± 1423.73 |
| Hopper-v5 | 3395.56 ± 50.64 | 3542.2 ± 51.5 | 2325.547 ± 1129.676 | ~3150 | 2310.46 ± 342.82 |
| Walker2d-v5 | 4014.48 ± 362.61 | 5007.0 ± 251.5 | 3863.203 ± 254.347 | ~4250 | 3591.45 ± 911.33 |
| Ant-v5 | 5150.34 ± 391.48 | 5850.2 ± 475.7 | 4615.791 ± 1354.111 | ~3980 | - |
| Humanoid-v5 | 5184.76 ± 273.78 | 5488.5 ± 81.2 | - | - | 4996.29 ± 686.40 |

> Training logs: [WandB - SAC MuJoCo](https://wandb.ai/placeholder/rlfromscratch-sac-mujoco)

#### TD3 (1M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | 10292.09 ± 926.49 | 10201.2 ± 772.8 | 9655.666 ± 969.916 | ~9750 | 9583.22 ± 126.09 |
| Hopper-v5 | 3307.63 ± 168.84 | 3472.2 ± 116.8 | 3606.390 ± 4.027 | ~2860 | 3134.61 ± 360.18 |
| Walker2d-v5 | 3709.81 ± 759.84 | 3982.4 ± 274.5 | 4717.823 ± 46.303 | ~4000 | 4057.59 ± 658.78 |
| Ant-v5 | 4456.58 ± 797.66 | 5116.4 ± 799.9 | 5813.274 ± 589.773 | ~3800 | - |
| Humanoid-v5 | 5088.79 ± 85.55 | 5189.5 ± 178.5 | - | - | 5035.36 ± 21.67 |

> Training logs: [WandB - TD3 MuJoCo](https://wandb.ai/placeholder/rlfromscratch-td3-mujoco)

#### PPO (1M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | 4859.36 ± 676.40 | 5783.9 ± 1244.0 | **5819.099 ± 663.530** | ~1670 | 1442.64 ± 46.03 |
| Hopper-v5 | **3352.50 ± 231.41** | 2609.3 ± 700.8 | 2410.435 ± 10.026 | ~1850 | 2382.86 ± 271.74 |
| Walker2d-v5 | **4273.01 ± 706.89** | 3588.5 ± 756.6 | 3478.798 ± 821.708 | ~1230 | 2287.95 ± 571.78 |
| Ant-v5 | 3091.34 ± 475.25 | **3258.4 ± 1079.3** | 1327.158 ± 451.577 | ~650 | - |
| Humanoid-v5 | **2130.14 ± 1099.33** | 787.1 ± 193.5 | - | - | 716.11 ± 49.08 |

> Training logs: [WandB - PPO MuJoCo 1M](https://wandb.ai/placeholder/rlfromscratch-ppo-mujoco-1m)

#### PPO (3M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | 6072.92 ± 735.34 | **7337.4 ± 1508.2** | - | ~3130 | - |
| Hopper-v5 | **3451.79 ± 132.82** | 3127.7 ± 413.0 | - | ~2460 | - |
| Walker2d-v5 | **5181.44 ± 673.34** | 4895.6 ± 704.3 | - | ~2600 | - |
| Ant-v5 | **4516.22 ± 358.72** | 4079.3 ± 880.2 | - | ~3000 | - |
| Humanoid-v5 | **5710.81 ± 391.43** | 1359.7 ± 572.7 | - | - | - |

> Training logs: [WandB - PPO MuJoCo 3M](https://wandb.ai/placeholder/rlfromscratch-ppo-mujoco-3m)

#### TRPO (1M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | **5115.75 ± 987.49** | 4471.2 ± 804.9 | 1785.476 ± 68.672 | ~850 | - |
| Hopper-v5 | 2991.73 ± 417.40 | 2046.0 ± 1037.9 | **3618.386 ± 356.768** | ~1200 | - |
| Walker2d-v5 | 4393.47 ± 436.21 | 3826.7 ± 782.7 | **4933.148 ± 1452.538** | ~600 | - |
| Ant-v5 | 2554.14 ± 448.38 | 2866.7 ± 707.9 | **4982.301 ± 663.761** | ~150 | - |
| Humanoid-v5 | 606.53 ± 99.35 | **810.1 ± 126.1** | - | - | - |

> Training logs: [WandB - TRPO MuJoCo 1M](https://wandb.ai/placeholder/rlfromscratch-trpo-mujoco-1m)

#### TRPO (3M steps)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah-v5 | **6489.54 ± 1008.39** | - | - | - | - |
| Hopper-v5 | **3439.49 ± 143.29** | - | - | - | - |
| Walker2d-v5 | **5070.51 ± 562.33** | - | - | - | - |
| Ant-v5 | **4724.01 ± 301.08** | - | - | - | - |
| Humanoid-v5 | **4280.01 ± 960.29** | - | - | - | - |

> Training logs: [WandB - TRPO MuJoCo 3M](https://wandb.ai/placeholder/rlfromscratch-trpo-mujoco-3m)

### Atari Performance Comparison

#### DQN (10M frames)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| PongNoFrameskip-v4 | **20.91 ± 0.04** | 20.2 ± 2.3 | 20.602 ± 0.613 | - | 20.25 ± 0.41 |
| BreakoutNoFrameskip-v4 | **380.60 ± 14.57** | 133.5 ± 44.6 | 358.327 ± 61.981 | - | 366.928 ± 39.89 |
| SpaceInvadersNoFrameskip-v4 | **1407.77 ± 256.20** | 947.9 ± 155.3 | 622.742 ± 201.564 | - | - |
| QbertNoFrameskip-v4 | 11546.90 ± 1749.36 | **11620.2 ± 786.1** | 9496.774 ± 5399.633 | - | - |
| SeaquestNoFrameskip-v4 | **7142.96 ± 1024.18** | 3213.9 ± 381.6 | 2000.290 ± 606.644 | - | - |
| BeamRiderNoFrameskip-v4 | **8029.93 ± 742.83** | - | 4295.946 ± 1790.458 | - | 6673.24 ± 1434.37 |
| FreewayNoFrameskip-v4 | **33.82 ± 0.04** | - | - | - | - |

> **DQN Atari note:** Pong uses `n-step=1`. For Pong and Freeway, a small number of seeds with exceptionally poor performance are excluded from the reported results.

> Training logs: [WandB - DQN Atari 10M](https://wandb.ai/placeholder/rlfromscratch-dqn-atari-10m)

#### PPO (10M frames)

| Environment | RLfromScratch | [Tianshou](https://github.com/thu-ml/tianshou) | [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Spinning Up](https://spinningup.openai.com/) | [CleanRL](https://github.com/vwxyzjn/cleanrl) |
| --- | --- | --- | --- | --- | --- |
| PongNoFrameskip-v4 | 20.09 ± 0.48 | 20.3 ± 1.2 | **20.989 ± 0.105** | - | 20.36 ± 0.20 |
| BreakoutNoFrameskip-v4 | **428.98 ± 21.51** | 283.0 ± 74.3 | 398.033 ± 33.328 | - | 414.66 ± 28.09 |
| SpaceInvadersNoFrameskip-v4 | 1012.52 ± 198.44 | **1641.3** | 960.331 ± 425.355 | - | - |
| QbertNoFrameskip-v4 | **15758.40 ± 1016.87** | 12341.8 ± 1760.7 | 15627.108 ± 3313.538 | - | - |
| SeaquestNoFrameskip-v4 | 1544.16 ± 412.53 | 1035.2 ± 353.6 | **1783.636 ± 34.096** | - | - |
| BeamRiderNoFrameskip-v4 | 2717.20 ± 389.58 | - | **3397.000 ± 1662.368** | - | 1915.93 ± 484.58 |
| FreewayNoFrameskip-v4 | 32.83 ± 0.33 | - | - | - | - |

> Training logs: [WandB - PPO Atari 10M](https://wandb.ai/placeholder/rlfromscratch-ppo-atari-10m)

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Training

```bash
# DQN on Atari
python run_dqn.py env.name=PongNoFrameskip-v4

# PPO on MuJoCo
python run_ppo.py env.name=HalfCheetah-v4

# PPO on Atari
python run_ppo_atari.py env.name=PongNoFrameskip-v4

# SAC on MuJoCo
python run_sac.py env.name=HalfCheetah-v4

# TD3 on MuJoCo
python run_td3.py env.name=HalfCheetah-v4

# TRPO on MuJoCo
python run_trpo.py env.name=HalfCheetah-v4
```

### Batch Experiments

```bash
# Run PPO on all MuJoCo environments
bash scripts/ppo_mujoco.sh

# Run DQN on multiple Atari games
bash scripts/dqn_multi_games.sh

# Run SAC on all MuJoCo environments
bash scripts/sac_mujoco.sh

# Run TD3 on all MuJoCo environments
bash scripts/td3_mujoco.sh
```

## Key Features

- Modular design: algorithms, environments, agents, and replay buffers are fully decoupled
- Hydra-based configuration management with command-line overrides and parallel experiments
- WandB and TensorBoard logging support
- DQN supports multiple variants: Double DQN, Dueling DQN, Prioritized Experience Replay, N-step Returns
- On-Policy algorithms support GAE, Observation Normalization, and Reward Normalization
- SB3 vectorized environments (SubprocVecEnv) for parallel sampling

## Tech Stack

- PyTorch 2.x
- Gymnasium
- Hydra / OmegaConf
- Stable Baselines3 (vectorized environment wrappers)
- WandB / TensorBoard
