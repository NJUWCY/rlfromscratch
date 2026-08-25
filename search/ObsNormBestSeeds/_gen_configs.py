"""Generate best-seed sweep configs for SAC / TD3 / DQN / PPO-atari.

Hyper-parameters are copied verbatim from the ``.hydra/overrides.yaml`` of the
exact expert source runs referenced in
``scripts/collect/collect_newest_100eps.sh`` (with seed / env.name / log_dir
promoted into ``search_space``). PPO-mujoco and TRPO-mujoco configs already
exist and are left untouched.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def write(name: str, spec: dict) -> None:
    (HERE / name).write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", name)


# --- SAC / MuJoCo -----------------------------------------------------------
SAC_FIXED = [
    "algorithm=sac",
    "train_action_deterministic=false",
    "env=mujoco",
    "env.num_training_envs=1",
    "save_interval=100000",
    "test_interval=10000",
    "train_log_interval=10000",
    "total_epoch=1500000",
    "algorithm.buffer_size=1000000",
    "algorithm.learn_temp=false",
    "algorithm.actor_lr=3e-4",
    "algorithm.critic_lr=3e-4",
    "interact_per_epoch=1",
    "algorithm.init_temp=0.2",
    "experiment_name=SAC-${env.name}",
]
SAC = {
    "Ant-v5": 1, "Hopper-v5": 4, "HalfCheetah-v5": 5,
    "Walker2d-v5": 5, "Humanoid-v5": 8,
}
for env, seed in SAC.items():
    short = env.split("-")[0].lower()
    write(f"sac_{short}_seed{seed}.json", {
        "run_script": "run_sac.py",
        "sweep_name": "SAC-best-seed",
        "fixed_overrides": SAC_FIXED,
        "search_space": {"seed": [seed], "env.name": [env]},
    })

# --- TD3 / MuJoCo -----------------------------------------------------------
TD3_FIXED = [
    "algorithm=td3",
    "train_action_deterministic=false",
    "env=mujoco",
    "env.num_training_envs=1",
    "save_interval=100000",
    "test_interval=10000",
    "train_log_interval=10000",
    "total_epoch=1500000",
    "algorithm.buffer_size=1000000",
    "interact_per_epoch=1",
    "start_train_step=25000",
    "experiment_name=TD3-${env.name}",
]
TD3 = {
    "Ant-v5": 5, "Hopper-v5": 6, "HalfCheetah-v5": 5,
    "Walker2d-v5": 5, "Humanoid-v5": 9,
}
for env, seed in TD3.items():
    short = env.split("-")[0].lower()
    write(f"td3_{short}_seed{seed}.json", {
        "run_script": "run_td3.py",
        "sweep_name": "TD3-best-seed",
        "fixed_overrides": TD3_FIXED,
        "search_space": {"seed": [seed], "env.name": [env]},
    })

# --- DQN / Atari ------------------------------------------------------------
DQN_FIXED = [
    "algorithm=dqn",
    "train_action_deterministic=true",
    "env=atari",
    "env.num_training_envs=1",
    "save_interval=100000",
    "test_interval=10000",
    "train_log_interval=5000",
    "total_epoch=2500000",
    "start_train_step=20000",
    "algorithm.buffer_size=1000000",
    "algorithm.huber_loss=true",
    "algorithm.end_epsilon=0.01",
    "algorithm.target_update_interval=2000",
    "algorithm.target_update_tau=1",
    "algorithm.epsilon_timestep=250000",
    "experiment_name=DQN-${env.name}",
]
# env: (seed, nstep). run_dir key order = seed, env.name, algorithm.nstep so the
# directory name reproduces "seed-<s>_name-<env>_nstep-<n>".
DQN = {
    "BreakoutNoFrameskip-v4": (5, 1),
    "SeaquestNoFrameskip-v4": (8, 1),
    "QbertNoFrameskip-v4": (7, 1),
    "SpaceInvadersNoFrameskip-v4": (3, 1),
    "BeamRiderNoFrameskip-v4": (5, 1),
    "FreewayNoFrameskip-v4": (7, 1),
    "PongNoFrameskip-v4": (0, 3),
}
for env, (seed, nstep) in DQN.items():
    short = env.replace("NoFrameskip-v4", "").lower()
    write(f"dqn_{short}_seed{seed}.json", {
        "run_script": "run_dqn.py",
        "sweep_name": "DQN-best-seed",
        "fixed_overrides": DQN_FIXED,
        "search_space": {
            "seed": [seed],
            "env.name": [env],
            "algorithm.nstep": [nstep],
        },
    })

# --- PPO / Atari ------------------------------------------------------------
PPO_ATARI_FIXED = [
    "algorithm=ppo",
    "env=atari",
    "env.num_training_envs=8",
    "total_epoch=10000",
    "interact_per_epoch=128",
    "update_step_per_epoch=1",
    "train_log_interval=10",
    "save_interval=1000",
    "test_interval=50",
    "train_action_deterministic=false",
    "algorithm.collect_traj=false",
    "algorithm.buffer_name=ReplayBuffer",
    "algorithm.update_epochs=4",
    "algorithm.minibatch_size=256",
    "algorithm.use_grad_clip=true",
    "algorithm.max_grad_norm=0.5",
    "algorithm.advan_norm=true",
    "algorithm.return_scaling=false",
    "algorithm.actor_lr=2.5e-4",
    "algorithm.critic_lr=2.5e-4",
    "algorithm.encoder_lr=2.5e-4",
    "algorithm.lr_decay=true",
    "algorithm.entropy_coef=0.01",
    "algorithm.eps_clip=0.1",
    "algorithm.value_clip=0.1",
    "algorithm.common_head=true",
    "algorithm.initialize=true",
    "env.scale=true",
    "env.obs_norm=false",
    "algorithm.test_epsilon=0.01",
    "experiment_name=PPO-atari-${env.name}",
]
PPO_ATARI = {
    "BreakoutNoFrameskip-v4": 3,
    "SeaquestNoFrameskip-v4": 5,
    "QbertNoFrameskip-v4": 9,
    "SpaceInvadersNoFrameskip-v4": 1,
    "BeamRiderNoFrameskip-v4": 6,
    "PongNoFrameskip-v4": 8,
    "FreewayNoFrameskip-v4": 1,
}
for env, seed in PPO_ATARI.items():
    short = env.replace("NoFrameskip-v4", "").lower()
    write(f"ppo_atari_{short}_seed{seed}.json", {
        "run_script": "run_ppo_atari.py",
        "sweep_name": "PPO-atari-best-seed",
        "fixed_overrides": PPO_ATARI_FIXED,
        "search_space": {"seed": [seed], "env.name": [env]},
    })

print("done")
