from dbench.capability_frontier import CAPABILITY_FRONTIER, task_by_id, validate_frontier


def test_capability_frontier_is_unique_hard_and_buildable():
    assert len({x.id for x in CAPABILITY_FRONTIER}) == len(CAPABILITY_FRONTIER)
    assert len({x.family for x in CAPABILITY_FRONTIER}) == len(CAPABILITY_FRONTIER)
    assert all(x.hard for x in CAPABILITY_FRONTIER)
    report = validate_frontier(seed=7)
    assert report["passes"], report


def test_frontier_lookup_refuses_unknown_task_ids():
    assert task_by_id("k5_invisible").family == "invisible_k5"
    try:
        task_by_id("post_hoc_winner")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown task ID was accepted")
