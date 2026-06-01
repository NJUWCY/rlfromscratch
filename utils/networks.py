import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np 
from gymnasium.spaces import Space 
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import AffineTransform, TanhTransform

class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], num_actions):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()
        C, H, W = input_shape

        self.conv1 = nn.Conv2d(C, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        
        with torch.no_grad():  
            dummy = torch.zeros(1, *input_shape)  
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            linear_input_size = x.view(1, -1).size(1)  

        self.fc1 = nn.Linear(linear_input_size, 256)
        
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x


# This network is used in TRPO
class MLPNetwork(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_sizes=[30,], activation=nn.ReLU):
        super(MLPNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(activation())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)
    

class Actor(nn.Module): # This is designed in TRPO, in other algorithm there are more to implement
    def __init__(self, observation_space:Space, action_space:Space):
        super(Actor, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self,x):
        raise NotImplementedError

    def get_action(self, states:np.ndarray,deterministic:bool):
        raise NotImplementedError

    def get_dist(self, states:np.ndarray):
        raise NotImplementedError

    def get_log_prob(self, states:np.ndarray, actions:np.ndarray):
        raise NotImplementedError


class GaussianActor(Actor):
    # TODO: add image input class
    """
    use network to compute the mean and standard deviation of the Gaussian distribution
    """
    def __init__(self, observation_space:Space, action_space:Space, net_architecture:nn.Module,device="cpu", clip_sigma=True, action_eps=1e-6, rescale=True, action_bound_method="tanh", **kwargs):
        super(GaussianActor, self).__init__(observation_space, action_space)

        self.mu = net_architecture(observation_space.shape[0], action_space.shape[0], **kwargs)
        # you can use network to compute the log_sigma according to the state
        self.log_sigma =  nn.Parameter(torch.zeros(size=(1,action_space.shape[0]))) 
        self.clip_sigma = clip_sigma
        self.action_eps = action_eps
        self.device = device

        self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
        self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2
        self.rescale = rescale
        self.action_bound_method = action_bound_method


    def forward(self,x:torch.Tensor):
        mu = self.mu(x)
        log_sigma = self.log_sigma
        if self.clip_sigma:
            log_sigma = log_sigma.clamp(min=-5, max=2)
        sigma = torch.exp(log_sigma).expand_as(mu)
        return mu, sigma
    
    def _action_clamp(self,actions:torch.Tensor):
        if self.rescale:
            return torch.clamp(actions,min=self.low+self.action_eps,max=self.high-self.action_eps)
        return actions

    
    def get_action(self, states:torch.Tensor,deterministic:bool):
        """
        select_action 的 Docstring
        :param states: (batch, state_dim)
        :return: actions: (batch, action_dim)
        """
        dist = self.get_dist(states) 
        
        if deterministic:
            # take the mean of the original Gaussian distribution and transform it 
            if self.rescale:
                if self.action_bound_method=="tanh":
                    actions = self.scale*torch.tanh(dist.base_dist.mean) + self.loc
                else:
                    raise NotImplementedError 
            else:
                actions = dist.base_dist.mean

        else:
            actions = dist.sample()
        
        # clamp the action to avoid the log_prob=nan
        actions = self._action_clamp(actions)
        log_probs = dist.log_prob(actions)
        return actions.detach().cpu().numpy(), {"log_probs":log_probs.detach().cpu().numpy()}
    
    def resample_action(self, states:torch.Tensor):
        """
        resample the action from the distribution, return the action which is differentiable
        """
        # Here we use the reparameterization trick to sample the action, which is used in SAC, the transformation in get_dist will have the following problem:
        # If the actions=1, then the log_prob will calculate nan.  
        assert self.action_bound_method=="tanh", "Only tanh transformation is supported in resample_action function"
        mu,std = self.forward(states)
        dist = Independent(Normal(mu,std), 1) # this will be seen as (batch,) action_dim-dimension distributions instead of batch*action_dim 1-dimension distributions
        u = dist.rsample()
        actions = torch.tanh(u)*self.scale + self.loc
        log_probs = dist.log_prob(u) - torch.log(1-torch.tanh(u).pow(2)).sum(-1, keepdim=True) # the log_prob of the action after transformation
        return actions, log_probs

    def get_dist(self, states:torch.Tensor):
        mu,std = self.forward(states)
        dist = Independent(Normal(mu,std), 1) # this will be seen as (batch,) action_dim-dimension distributions instead of batch*action_dim 1-dimension distributions
        if self.rescale:
            if self.action_bound_method=="tanh":
                dist = TransformedDistribution(dist, [TanhTransform(cache_size=1), AffineTransform(loc=self.loc, scale=self.scale)]) # The notation wrote by torch is wrong: the order of the transformation is tanh->affine; but the notation is affine->tanh
            else:
                raise NotImplementedError 
        return dist
    
    def get_log_prob(self, states:torch.Tensor, actions:torch.Tensor):
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions)
        return log_probs


class Critic(nn.Module):
    def __init__(self,observation_space:Space, action_space:Space=None):
        super(Critic, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self,x):
        raise NotImplementedError


class ValueFunction(Critic):
    def __init__(self, observation_space:Space, net_architecture:nn.Module, **kwargs):
        super(ValueFunction, self).__init__(observation_space)

        self.net = net_architecture(observation_space.shape[0],1, **kwargs)
    
    def forward(self,states:torch.Tensor):
        return self.net(states)


class QFunction(Critic):
    def __init__(self, observation_space:Space, action_space:Space, net_architecture:nn.Module, **kwargs):
        super(QFunction, self).__init__(observation_space, action_space)

        self.net = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)
    
    def forward(self,states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: (batch,1)
        """
        inputs = torch.concat([states, actions], dim=1)
        return self.net(inputs)

class DoubleQFunction(QFunction):
    def __init__(self, observation_space:Space, action_space:Space, net_architecture:nn.Module, **kwargs):
        super(DoubleQFunction, self).__init__(observation_space, action_space, net_architecture, **kwargs)

        self.net1 = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)
        self.net2 = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)

    def forward(self,states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: (2,batch,1)
        """
        inputs = torch.cat([states, actions], dim=1)
        return self.net1(inputs), self.net2(inputs)