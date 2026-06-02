from memory.memory import ReplayBuffer, TrajectoryRollout
from memory.expert_dataset import ExpertDataset, load_expert_data

BUFFER_DICT = {
    "ReplayBuffer": ReplayBuffer,
    "TrajectoryRollout": TrajectoryRollout,
    "PrioritizedReplayBuffer": None, # you can implement PrioritizedReplayBuffer by yourself and add it to this dict
}
