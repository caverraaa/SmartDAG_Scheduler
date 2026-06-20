"""Two-head autoregressive policy + critic (TZ §6.3).

pi(a) = pi(task | s) * pi(node | s, task). Head 1 scores ready tasks, Head 2
scores alive nodes given the chosen task. Masking sets non-candidate logits to
-inf (exactly zero probability). Variable candidate-set sizes are handled by
per-instance scoring — never a fixed Discrete action space.
"""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from src.env.observation import Observation
from src.rl.gnn_encoder import GNNEncoder
from src.rl.obs_tensors import ObsTensors, obs_to_tensors

_NEG_INF = float("-inf")


class TwoHeadPolicy(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden: int = 64, glob_in: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        ctx_dim = 2 * hidden + glob_in  # [g, c, globals]
        self.head_task = nn.Sequential(
            nn.Linear(hidden + ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.head_node = nn.Sequential(
            nn.Linear(hidden + hidden + ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, t: ObsTensors) -> tuple[Tensor, Tensor, Tensor]:
        h, g, n_emb, c = self.encoder(t.task_features, t.edge_index, t.node_features)
        ctx = torch.cat([g, c, t.globals], dim=-1)
        return h, n_emb, ctx

    def task_logits(self, h: Tensor, ctx: Tensor, ready_mask: Tensor) -> Tensor:
        ctx_b = ctx.unsqueeze(0).expand(h.shape[0], -1)
        scores = self.head_task(torch.cat([h, ctx_b], dim=-1)).squeeze(-1)
        return scores.masked_fill(~ready_mask, _NEG_INF)

    def node_logits(self, h_tau: Tensor, n_emb: Tensor, ctx: Tensor, alive_mask: Tensor) -> Tensor:
        cond = torch.cat([h_tau, ctx], dim=-1).unsqueeze(0).expand(n_emb.shape[0], -1)
        scores = self.head_node(torch.cat([n_emb, cond], dim=-1)).squeeze(-1)
        return scores.masked_fill(~alive_mask, _NEG_INF)

    def value(self, ctx: Tensor) -> Tensor:
        return self.critic(ctx).squeeze(-1)

    def act(self, obs: Observation) -> tuple[tuple[int, int], Tensor, Tensor]:
        t = obs_to_tensors(obs)
        h, n_emb, ctx = self.encode(t)
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        task_id = task_dist.sample()
        node_dist = Categorical(logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask))
        node_id = node_dist.sample()
        log_prob = task_dist.log_prob(task_id) + node_dist.log_prob(node_id)
        return (int(task_id), int(node_id)), log_prob, self.value(ctx)

    def evaluate_action(
        self, obs: Observation, task_id: int, node_id: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = obs_to_tensors(obs)
        h, n_emb, ctx = self.encode(t)
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        node_dist = Categorical(logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask))
        log_prob = task_dist.log_prob(torch.tensor(task_id)) + node_dist.log_prob(
            torch.tensor(node_id)
        )
        # Use unmasked logits for entropy so gradients flow through head_task even
        # when only one task is ready (masked entropy collapses to a constant 0 in
        # that case, killing the gradient path through head_task).
        all_ready = torch.ones_like(t.ready_mask)
        all_alive = torch.ones_like(t.alive_mask)
        task_entropy = Categorical(logits=self.task_logits(h, ctx, all_ready)).entropy()
        node_entropy = Categorical(
            logits=self.node_logits(h[task_id], n_emb, ctx, all_alive)
        ).entropy()
        entropy = task_entropy + node_entropy
        return log_prob, entropy, self.value(ctx)
