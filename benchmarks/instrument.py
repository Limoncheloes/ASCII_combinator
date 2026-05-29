"""Stage timing helper.

The bench harness wraps each pipeline stage with `stage(name)` and reads
totals/medians from a `StageRegistry`. Source modules in `ascii_combinator/`
are NOT instrumented — wrapping happens in `benchmarks/run.py`.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageRegistry:
    _samples: defaultdict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, name: str, elapsed_s: float) -> None:
        self._samples[name].append(elapsed_s)

    def samples_for(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-stage median seconds and share-% of total sample time.

        share_pct is computed against sum-of-all-samples (not sum-of-medians) so
        that stages invoked many times per iteration get credit proportional to
        their wall-clock contribution.
        """
        non_empty = {
            name: samples
            for name, samples in self._samples.items()
            if samples
        }
        totals = {name: sum(samples) for name, samples in non_empty.items()}
        grand_total = sum(totals.values()) or 1.0
        return {
            name: {
                "median_s": round(statistics.median(samples), 6),
                "share_pct": round(100.0 * totals[name] / grand_total, 2),
                "count": len(samples),
            }
            for name, samples in non_empty.items()
        }


_DEFAULT_REGISTRY = StageRegistry()


@contextmanager
def stage(name: str, registry: StageRegistry | None = None):
    """Time the wrapped block and append the elapsed seconds to the registry."""
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    t0 = time.perf_counter()
    try:
        yield
    finally:
        reg.record(name, time.perf_counter() - t0)
