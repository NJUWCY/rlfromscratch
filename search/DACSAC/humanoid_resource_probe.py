"""CPU-only resident-memory probe: full replay + expert + eight eval envs.

Does not train, initialize CUDA or contact SwanLab. PSS includes the process tree.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import torch
from hydra import compose, initialize_config_dir
from env.make_envs import make_vec_envs
from memory.memory import AbsorbingReplayBuffer, ExpertDataset
from humanoid_wave1 import FIXED

def snapshot(stage):
    processes=[psutil.Process()]+psutil.Process().children(recursive=True)
    totals={'rss':0,'pss':0,'uss':0}
    for proc in processes:
        try:
            m=proc.memory_full_info()
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
        for key in totals:
            totals[key]+=getattr(m,key,0)
    print(json.dumps({'stage':stage,'process_count':len(processes),
        **{key+'_gib':value/2**30 for key,value in totals.items()}},indent=2),flush=True)

def main():
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(ROOT/'config'),version_base='1.3'):
        cfg=compose(config_name='config',overrides=FIXED)
    training=make_vec_envs(cfg.env,True,seed=0)
    testing=None
    try:
        snapshot('imports_and_one_training_env')
        replay=AbsorbingReplayBuffer(training.observation_space,training.action_space,
                                   1000000,gamma=cfg.gamma,nstep=1,num_envs=1)
        exact=sum(v.nbytes for v in replay.buffer.values())
        print(json.dumps({'replay_bytes':exact,'replay_gib':exact/2**30}),flush=True)
        for array in replay.buffer.values():
            array.fill(0)  # commit otherwise lazy numpy allocation to RAM
        snapshot('full_replay_committed')
        expert=ExpertDataset(cfg.expert_dataset.data_path,training.observation_space,
                            training.action_space,5,1,cfg.gamma,1,absorbing=True,
                            random_start=False)
        print('expert_buffer_size',expert.buffer_size,flush=True)
        snapshot('replay_and_expert_loaded')
        testing=make_vec_envs(cfg.env,False,seed=10000)
        observations=testing.reset()
        testing.step(np.zeros((8,17),dtype=np.float32))
        snapshot('full_replay_plus_eight_eval_envs')
        print(json.dumps({'eval_observation_shape':list(observations.shape),
                         'note':'CPU probe excludes CUDA runtime, optimizers, SwanLab and training minibatches.'}),flush=True)
    finally:
        if testing is not None:
            testing.close()
        training.close()

if __name__=='__main__':
    main()
