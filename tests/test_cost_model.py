from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass
from src.env.cost_model import comm_time, energy, exec_time, make_speed_table
from src.utils.seeding import make_rng


def _node(speed: float, power: float = 100.0, bw: float = 10.0) -> ComputeNode:
    return ComputeNode(
        node_id=0,
        node_type=NodeType.CPU,
        speed_by_class={tc: speed for tc in TaskClass},
        power_w=power,
        bandwidth=bw,
    )


def test_exec_time_and_energy() -> None:
    t = Task(0, 8.0, 1.0, TaskClass.SEQUENTIAL)
    n = _node(speed=2.0, power=150.0)
    assert exec_time(t, n) == 4.0
    assert energy(t, n) == 600.0


def test_comm_time() -> None:
    assert comm_time(20.0, 10.0) == 2.0
    assert comm_time(20.0, 10.0, latency=0.5) == 2.5


def test_speed_table_beta_ratio() -> None:
    # Test with beta=5.0
    table = make_speed_table(make_rng(0), beta=5.0)
    assert set(table.keys()) == set(NodeType)
    for tc in TaskClass:
        speeds = [table[nt][tc] for nt in NodeType]
        ratio = max(speeds) / min(speeds)
        assert 4.0 <= ratio <= 6.5  # jitter-aware window for beta=5.0

    # Test with beta=2.0 to verify beta drives the ratio
    table2 = make_speed_table(make_rng(1), beta=2.0)
    assert set(table2.keys()) == set(NodeType)
    for tc in TaskClass:
        speeds = [table2[nt][tc] for nt in NodeType]
        ratio = max(speeds) / min(speeds)
        assert 1.5 <= ratio <= 2.7  # jitter-aware window for beta=2.0


def test_speed_table_affinity() -> None:
    table = make_speed_table(make_rng(1), beta=5.0)

    # data_parallel fastest on GPU or TPU
    dp = {nt: table[nt][TaskClass.DATA_PARALLEL] for nt in NodeType}
    best = max(dp, key=dp.get)
    assert best in (NodeType.GPU, NodeType.TPU)

    # sequential fastest on CPU
    seq = {nt: table[nt][TaskClass.SEQUENTIAL] for nt in NodeType}
    best = max(seq, key=seq.get)
    assert best == NodeType.CPU

    # streaming fastest on FPGA
    stream = {nt: table[nt][TaskClass.STREAMING] for nt in NodeType}
    best = max(stream, key=stream.get)
    assert best == NodeType.FPGA
