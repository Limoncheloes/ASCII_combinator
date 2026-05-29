import time

from benchmarks.instrument import StageRegistry, stage


def test_stage_records_elapsed_time():
    reg = StageRegistry()
    with stage("foo", registry=reg):
        time.sleep(0.01)
    samples = reg.samples_for("foo")
    assert len(samples) == 1
    assert samples[0] >= 0.01


def test_multiple_samples_accumulate():
    reg = StageRegistry()
    for _ in range(3):
        with stage("bar", registry=reg):
            pass
    assert len(reg.samples_for("bar")) == 3


def test_summary_returns_median_and_share():
    reg = StageRegistry()
    with stage("a", registry=reg):
        time.sleep(0.02)
    with stage("b", registry=reg):
        time.sleep(0.01)
    summary = reg.summary()
    assert "a" in summary and "b" in summary
    assert summary["a"]["median_s"] > summary["b"]["median_s"]
    assert 0.0 < summary["a"]["share_pct"] <= 100.0
    assert 0.0 < summary["b"]["share_pct"] <= 100.0
    total_share = summary["a"]["share_pct"] + summary["b"]["share_pct"]
    assert abs(total_share - 100.0) < 0.01


def test_unknown_stage_returns_empty_list():
    reg = StageRegistry()
    assert reg.samples_for("nope") == []
