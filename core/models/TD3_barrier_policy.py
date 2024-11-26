import numpy as np
import torch as th
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from gym import spaces
from functools import partial

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.td3.policies import Actor, TD3Policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp
)
from stable_baselines3.common.type_aliases import Schedule

from SRL.core.models.barrier import BarrierNetwork

class SafeTD3Policy(TD3Policy):
    """
    Safe TD3 policy class with actor-critic architecture and safety constraints.
    Incorporates barrier certificates for safe action selection.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            safety_margin: float = 0.1,
    ):
        # Sign of initialization
        self.initialization = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor
        )

        # barrier network parameters
        barrier_arch = get_actor_critic_barrier_arch(net_arch)[2] if net_arch else [64, 64]

        self.barrier_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "hidden_sizes": barrier_arch,
            "normalize_images": normalize_images,
        }

        self.barrier_net = None

        self.safety_margin = safety_margin

        self.initialization = True

        self._build(lr_schedule)



    def _build(self, lr_schedule: Schedule) -> None:
        if not self.initialization:
            return
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
            self.barrier_net = self.make_barrier(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)
            self.barrier_net = self.make_barrier(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)

        self.barrier_net.optimizer = self.optimizer_class(self.barrier_net.parameters(), lr=lr_schedule(1),
                                                     **self.optimizer_kwargs)


        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def make_barrier(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> BarrierNetwork:
        #barrier_kwargs = self._update_features_extractor(self.barrier_kwargs, features_extractor)
        return BarrierNetwork(**self.barrier_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        """Set training mode."""
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.barrier_net.set_training_mode(mode)
        self.training = mode

def get_actor_critic_barrier_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch, barrier_arch = net_arch, net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        assert "br" in net_arch, "Error: no key 'br' was provided in net_arch for the barrier network"
        actor_arch, critic_arch, barrier_arch = net_arch["pi"], net_arch["qf"], net_arch["br"]
    return actor_arch, critic_arch, barrier_arch
