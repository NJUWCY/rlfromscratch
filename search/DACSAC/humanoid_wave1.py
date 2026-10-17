"""Proposed first wave: 4 actor/critic LR pairs x 3 temperature modes.

Training starts only with --run; the default prints the reviewable plan.
Uses the repository's existing parallel_search.py, one group per temp mode.
"""
import argparse
import datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERT = ROOT / 'outputs/collected/SAC-Mujoco/sac_Humanoid-v5_newest_100eps_min_reward_0.hdf5'
FIXED = [
    'algorithm=dacsac', 'env=mujoco', 'env.name=Humanoid-v5',
    'env.num_training_envs=1', 'env.num_testing_envs=8',
    'env.obs_norm=false', 'env.absorbing=true', 'seed=0', 'gamma=0.99',
    'total_epoch=1500000', 'interact_per_epoch=1', 'update_step_per_epoch=1',
    'update_discriminator_step_per_epoch=1', 'discriminator_train_interval=1',
    'start_train_step=1000', 'test_interval=10000', 'test_episodes=10',
    'train_log_interval=1000', 'save_interval=100000',
    'train_action_deterministic=false', 'use_swanlab=true',
    'use_tensorboard=false', 'use_wandb=false',
    'algorithm.rl.buffer_size=1000000', 'algorithm.rl.batch_size=256',
    'algorithm.rl.hidden_sizes=[256,256]', 'algorithm.rl.activation=relu',
    'algorithm.rl.actor_optimizer=Adam', 'algorithm.rl.critic_optimizer=Adam',
    'algorithm.rl.temp_optimizer=Adam', 'algorithm.rl.temp_lr=3e-4',
    'algorithm.rl.init_temp=0.1', 'algorithm.rl.update_log_temp=true',
    'algorithm.rl.nstep=1', 'algorithm.rl.target_update_tau=0.01',
    'algorithm.rl.critic_loss_coef=1.0',
    'algorithm.discriminator.disc_lr=1e-4',
    'algorithm.discriminator.discriminator_optimizer=Adam',
    'algorithm.discriminator.discriminator_weight_decay=0.0',
    'algorithm.discriminator.discriminator_hidden_sizes=[256,256]',
    'algorithm.discriminator.discriminator_activation=tanh',
    'algorithm.discriminator.spectral_norm=true',
    'algorithm.discriminator.discriminator_gradient_penalty=true',
    'algorithm.discriminator.gradient_penalty_coef=10.0',
    'algorithm.discriminator.target_gradient_norm=1.0',
    'algorithm.discriminator.discriminator_train_steps=1',
    'algorithm.discriminator.discriminator_batch_size=256',
    'algorithm.discriminator.reward_function=AIRL',
    f'expert_dataset.data_path={EXPERT}', 'expert_dataset.trajectory_num=5',
    'expert_dataset.subsample_frequency=1', 'expert_dataset.random_start=false',
]
MODES = {
    'fixed_alpha_0.1': ['algorithm.rl.learn_temp=false', 'algorithm.rl.target_entropy=-0.5'],
    'auto_entropy_-0.5': ['algorithm.rl.learn_temp=true', 'algorithm.rl.target_entropy=-0.5'],
    'auto_entropy_-1.0': ['algorithm.rl.learn_temp=true', 'algorithm.rl.target_entropy=-1.0'],
}
LR = [0.0001, 0.0003]

def specs():
    return {name: {
        'run_script':'run_dac.py', 'sweep_name':'Humanoid-v5-DACSAC-wave1',
        'fixed_overrides':FIXED + mode,
        'search_space':{'algorithm.rl.actor_lr':LR, 'algorithm.rl.critic_lr':LR},
    } for name, mode in MODES.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    p.add_argument('--gpus', default='0,1', help='Physical GPU IDs, each listed once.')
    p.add_argument('--jobs-per-gpu', type=int, choices=(1,2,4), default=1)
    p.add_argument('--mode', choices=('offline','online'), default='offline')
    p.add_argument('--output', type=Path)
    p.add_argument('--epochs', type=int, default=1500000,
                   help='Use smaller values only for smoke/throughput checks, never convergence ranking.')
    p.add_argument('--timeout-hours', type=float, default=24)
    args=p.parse_args()
    if args.epochs<=0 or args.timeout_hours<=0:
        p.error('epochs and timeout-hours must be positive')
    gpu_ids=[s.strip() for s in args.gpus.split(',') if s.strip()]
    if not gpu_ids or len(set(gpu_ids))!=len(gpu_ids) or any(not s.isdigit() for s in gpu_ids):
        p.error('--gpus must contain unique nonnegative physical GPU IDs')
    slots=[gpu for _ in range(args.jobs_per_gpu) for gpu in gpu_ids]
    workers=min(4,len(slots))  # each temperature group has four candidates
    plan=[]
    group_specs=specs()
    for name,spec in group_specs.items():
        spec['fixed_overrides']=[f'total_epoch={args.epochs}' if x.startswith('total_epoch=') else x for x in spec['fixed_overrides']]
        for a,c in itertools.product(LR,LR):
            plan.append({'temperature_mode':name,'actor_lr':a,'critic_lr':c,'seed':0,
                         'epochs':args.epochs,'expert_trajectories':5,'subsample_frequency':1})
    print(json.dumps({'trials':plan,'workers':workers,'gpu_slots':slots,
                      'mode':args.mode,'total_env_steps':12*args.epochs},indent=2))
    if not (args.run or args.dry_run):
        return
    if args.run:
        import torch
        if not torch.cuda.is_available() or any(int(g)>=torch.cuda.device_count() for g in gpu_ids):
            p.error('Requested CUDA devices unavailable; no CPU training fallback.')
        if not EXPERT.is_file():
            p.error(f'Expert file missing: {EXPERT}')
        subprocess.check_call([sys.executable,str(ROOT/'search/DACSAC/humanoid_wave1_prepare.py')],cwd=ROOT)
    destination=args.output or ROOT/'outputs/Humanoid-v5-DACSAC-wave1'/datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    destination=destination.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (destination/'manifest.json').write_text(json.dumps({'created_at':datetime.datetime.now().astimezone().isoformat(),
        'git_commit':commit,'trials':plan,'swanlab_mode':args.mode,'gpu_slots':slots},indent=2)+'\n')
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'
    env['SWANLAB_MODE']=args.mode
    env['RES_OPTIONS']=env.get('RES_OPTIONS','ndots:1')
    env['TQDM_MININTERVAL']='60'
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    failure=False
    started=datetime.datetime.now().astimezone().isoformat()
    for name,spec in group_specs.items():
        spec['fixed_overrides']=[
            'expert_dataset.data_path='+str(ROOT/'outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5')
            if x.startswith('expert_dataset.data_path=') else x for x in spec['fixed_overrides']]
        spec_file=destination/(name+'.json')
        spec_file.write_text(json.dumps(spec,indent=2)+'\n')
        cmd=[sys.executable,str(ROOT/'parallel_search.py'),'--config',str(spec_file),
             '--workers',str(workers),'--gpus',','.join(slots),
             '--log-dir',str(destination/name),'--timeout-hours',str(args.timeout_hours)]
        if args.dry_run:
            cmd.append('--dry-run')
        rc=subprocess.call(cmd,cwd=ROOT,env=env)
        failure = failure or rc!=0
        if rc!=0:
            print(f'Group {name} failed; inspect stdout.log before interpreting its score.',file=sys.stderr)
    (destination/'sweep_status.json').write_text(json.dumps({'started_at':started,
        'ended_at':datetime.datetime.now().astimezone().isoformat(),'dry_run':args.dry_run,'failed':failure},indent=2)+'\n')
    print('Sweep output:',destination)
    if failure:
        raise SystemExit(1)

if __name__=='__main__':
    main()
