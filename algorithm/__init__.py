from .basealgorithm import BaseAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .baseonpolicy import OnPolicyAlgorithm
from .offline import OfflineAlgorithm


from .dqn import DQN 
from .trpo import TRPO 
from .ppo import PPO
from .sac import SAC
from .bc import BehaviorCloning


ALGORITHM_DICT = {
    "DQN": DQN,
    "TRPO":TRPO,
    "PPO": PPO,
    "SAC": SAC,
    "BC": BehaviorCloning,
}
