from .basealgorithm import BaseAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .baseonpolicy import OnPolicyAlgorithm
from .offline import OfflineAlgorithm
from .baseailalgorithm import AILAlgorithm



from .dqn import DQN 
from .trpo import TRPO 
from .ppo import PPO
from .sac import SAC
from .bc import BehaviorCloning
from .td3 import TD3
from .gail import GAIL, GAILPPO, GAILTRPO
from .airl import AIRL, AIRLPPO, AIRLTRPO
from .dac import DAC, DACSAC


ALGORITHM_DICT = {
    "DQN": DQN,
    "TRPO":TRPO,
    "PPO": PPO,
    "SAC": SAC,
    "BC": BehaviorCloning,
    "TD3": TD3,
    "GAIL": GAIL,
    "GAILPPO": GAILPPO,
    "GAILTRPO": GAILTRPO,
    "AIRL": AIRL,
    "AIRLPPO": AIRLPPO,
    "AIRLTRPO": AIRLTRPO,
    "DAC": DACSAC,
    "DACSAC": DACSAC,
}
