from env.make_envs import make_env
from agent.agent import AtariDQNAgent
from utils.utils import get_best_device, atari_to_useful_action,set_seed



from omegaconf import OmegaConf
import torch 
import numpy as np
import pickle
import argparse

Reward_Thresholds = {
    "PongNoFrameskip-v4":21, 
    "BreakoutNoFrameskip-v4": 350,
}
Length_Thresholds = {
    "PongNoFrameskip-v4":0, 
    "BreakoutNoFrameskip-v4": 0,
}
FORMAT = [
    "pkl",
]

def get_args():
    parser = argparse.ArgumentParser()

    # env args
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--frame_skip", type=int, default=4)

    # other args
    parser.add_argument("--model_file_pth", type=str, required=True)
    parser.add_argument("--collect_episodes", type=int, default=10)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_format", type=str, default="pkl", choices=FORMAT)

    return parser.parse_args()


def satisfy_condition(env_name, reward, length):
    """
    if the reward and episode length satisfy the condition, this episode of experience will be saved
    """
    if reward>=Reward_Thresholds[env_name] and length>=Length_Thresholds[env_name]:
        return True 
    return False


args = get_args()

set_seed(args.seed)
device = get_best_device()

env_args = dict(
    name=args.env_name,
    frame_stack=args.frame_stack,
    frame_skip=args.frame_skip
)
# create environment
env_args = OmegaConf.create(env_args)
print(f"Creating environment {env_args.name}")
env = make_env(env_args,is_training=False)


# create agent and load parameters
agent = AtariDQNAgent(env.observation_space, env.action_space,device).to(device)
agent.load(args.model_file_pth)
all_states = []
all_actions = []
all_next_states = []
all_rewards = []
all_terminates = []
all_truncates = []




traj_states = []
traj_actions = []
traj_next_states = []
traj_rewards = []
traj_terminates = []
traj_truncates = []
episode_rewards = []

# interact with envs and collect dataset
agent.eval()
state,info = env.reset(seed=args.seed)
while len(episode_rewards)<args.collect_episodes:
    with torch.no_grad():
        action, action_infos = agent.select_action(state,deterministic=True).item() # test with deterministic policy
    next_state, reward, terminate, truncate, info = env.step(action)
    traj_states.append(state)
    traj_actions.append(action)
    traj_next_states.append(next_state)
    traj_rewards.append(reward)
    traj_terminates.append(terminate)
    traj_truncates.append(truncate)
    

    if 'episode' in info:
        episode_reward = info['episode']['r']
        if satisfy_condition(env_args.name, episode_reward, len(traj_states)):
            episode_rewards.append(episode_reward)
            all_states.append(np.array(traj_states,dtype=np.float32))
            all_actions.append(np.array(traj_actions,dtype=np.float32))
            all_next_states.append(np.array(traj_next_states,dtype=np.float32))
            all_rewards.append(np.array(traj_rewards, dtype=np.float32))
            all_terminates.append(np.array(traj_terminates,dtype=np.float32))
            all_truncates.append(np.array(traj_truncates,dtype=np.float32))


        traj_states = []
        traj_actions = []
        traj_next_states = []
        traj_rewards = []
        traj_terminates = []
        traj_truncates = []

    if terminate or truncate:
        state, info = env.reset()
    else:
        state = next_state
env.close()



print(f"Collect {args.collect_episodes} trajectories with rewards' mean: {np.mean(episode_rewards)}")

# save the dataset
trajs_saved_path = f"./{env_args.name}_{args.collect_episodes}.{args.save_format}"
if args.save_format=="pkl":
    all_data = dict(
        states=all_states,
        actions=all_actions,
        next_states=all_next_states,
        rewards=all_rewards,
        terminates=all_terminates,
        truncates=all_truncates
    )
    with open(trajs_saved_path,"wb") as f:
        pickle.dump(all_data,f)

else:
    raise ValueError("This type of saved dataset is not supported yet.")
