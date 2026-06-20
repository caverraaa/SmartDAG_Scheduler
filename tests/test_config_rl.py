from src.utils.config import load_config


def test_rl_hyperparameters_loaded() -> None:
    cfg = load_config("config.yaml")
    assert cfg.lr == 0.0003
    assert cfg.clip_eps == 0.2
    assert cfg.gae_lambda == 0.95
    assert cfg.ppo_epochs == 4
    assert cfg.minibatch_size == 32
    assert cfg.entropy_coef == 0.01
    assert cfg.value_coef == 0.5
    assert cfg.rollout_episodes == 4
    assert cfg.total_updates == 50
    assert cfg.max_grad_norm == 0.5
    assert cfg.gnn_hidden == 64
    assert cfg.gnn_layers == 2
