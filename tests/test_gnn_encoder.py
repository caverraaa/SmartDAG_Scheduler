import torch

from src.rl.gnn_encoder import GNNEncoder


def _inputs(n: int = 4, m: int = 2, edges=((0, 1), (0, 2), (1, 3), (2, 3))):
    task_features = torch.randn(n, 15)
    node_features = torch.randn(m, 9)
    edge_index = (
        torch.tensor(list(zip(*edges, strict=False)), dtype=torch.long)
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return task_features, edge_index, node_features


def test_encoder_output_shapes() -> None:
    enc = GNNEncoder(hidden=32, layers=2)
    tf, ei, nf = _inputs()
    h, g, n_emb, c = enc(tf, ei, nf)
    assert h.shape == (4, 32)
    assert g.shape == (32,)
    assert n_emb.shape == (2, 32)
    assert c.shape == (32,)


def test_encoder_handles_empty_edges() -> None:
    enc = GNNEncoder(hidden=16, layers=2)
    tf, _, nf = _inputs(n=1, m=1, edges=())
    h, g, n_emb, c = enc(tf, torch.zeros((2, 0), dtype=torch.long), nf)
    assert h.shape == (1, 16) and not torch.isnan(h).any()


def test_encoder_is_direction_sensitive() -> None:
    # reversing all edges should change the per-task embeddings (proves reverse
    # stream carries information distinct from forward)
    torch.manual_seed(0)
    enc = GNNEncoder(hidden=32, layers=2)
    tf, ei, nf = _inputs()
    h_fwd, *_ = enc(tf, ei, nf)
    h_rev, *_ = enc(tf, ei.flip(0), nf)
    assert not torch.allclose(h_fwd, h_rev, atol=1e-5)


def test_encoder_gradients_flow() -> None:
    enc = GNNEncoder(hidden=16, layers=2)
    tf, ei, nf = _inputs()
    h, g, n_emb, c = enc(tf, ei, nf)
    (g.sum() + c.sum()).backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(gr is not None and gr.abs().sum() > 0 for gr in grads)
