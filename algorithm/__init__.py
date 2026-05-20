from .basealgorithm import BaseAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .baseonpolicy import OnPolicyAlgorithm


from .dqn import DQN 
from .trpo import TRPO 
from .ppo import PPO
from .sac import SAC


ALGORITHM_DICT = {
    "DQN": DQN,
    "TRPO":TRPO,
    "PPO": PPO
}