"""Convert a numpy Observation into torch tensors for the policy (TZ §6.2).

Feature normalisation is already applied upstream in build_observation; this is
a pure dtype/layout conversion.
"""

from dataclasses import dataclass

import torch

from src.env.observation import Observation


@dataclass
class ObsTensors:
    task_features: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    globals: torch.Tensor
    ready_mask: torch.Tensor
    alive_mask: torch.Tensor


def obs_to_tensors(obs: Observation) -> ObsTensors:
    return ObsTensors(
        task_features=torch.from_numpy(obs.task_features).float(),
        node_features=torch.from_numpy(obs.node_features).float(),
        edge_index=torch.from_numpy(obs.edge_index).long(),
        globals=torch.from_numpy(obs.globals).float(),
        ready_mask=torch.from_numpy(obs.ready_mask),
        alive_mask=torch.from_numpy(obs.alive_mask),
    )
