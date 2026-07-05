from typing import NamedTuple

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    graph: GraphsTuple
    actions: Action  # (# of env, batchsize/time horizon, num_agents, action_dim)
    rewards: Reward #(# of env, batchsize)
    costs: Cost #(# of env, batchsize)
    dones: Done
    log_pis: Array
    next_graph: GraphsTuple

    @property
    def length(self) -> int:
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        return self.length * self.time_horizon
