def test_torch_and_pyg_importable() -> None:
    import torch
    from torch_geometric.nn import SAGEConv

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
    assert SAGEConv(4, 8) is not None
