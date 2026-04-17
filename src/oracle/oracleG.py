from typing import List

from constants.segmentation_defaults import (
    STSS_MAX_EPS_METERS,
    STSS_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from core.point import Point
from core.segment import Segment
from engines.stss_manual import STSSOracleManual
from engines.stss_sklearn import STSSOracleSklearn


class OracleG:
    """
    Offline Geometric Stop-Segmentation Oracle utilizing the STSS
    (Semantics-Based Trajectory Segmentation Simplification) algorithm.

    Acts as a thin orchestrator on top of the STSS engine; all density-based
    clustering logic lives in `engines/stss_sklearn.py` or `engines/stss_manual.py`.
    """

    def __init__(
        self,
        min_samples: int = STSS_MIN_SAMPLES,
        max_eps: float = STSS_MAX_EPS_METERS,
        min_duration_seconds: float = STSS_MIN_DURATION_SECONDS,
        backend: str = "sklearn",
    ):
        """
        Initialize the geometric (stop-segmentation) oracle.

        Args:
            min_samples: Core-point neighbourhood size for the density cluster.
            max_eps: Maximum neighbour distance in meters.
            min_duration_seconds: Minimum duration to keep a Stop.
            backend: "sklearn" (OPTICS) or "manual" (DBSCAN-like) STSS implementation.
        """
        if backend == "sklearn":
            self.segmenter = STSSOracleSklearn(
                min_samples=min_samples,
                max_eps=max_eps,
                min_duration_seconds=min_duration_seconds,
            )
        elif backend == "manual":
            self.segmenter = STSSOracleManual(
                min_samples=min_samples,
                max_eps=max_eps,
                min_duration_seconds=min_duration_seconds,
            )
        else:
            raise ValueError(
                f"Unknown STSS backend '{backend}'. Expected 'sklearn' or 'manual'."
            )
        self.backend = backend

    def process(self, trajectory: List[Point]) -> List[Segment]:
        """
        Segment a full trajectory into Stop/Move segments using STSS.

        Args:
            trajectory: Ordered list of raw GPS points.

        Returns:
            List of Stop and Move segments in chronological order.
        """
        if not trajectory:
            return []

        return self.segmenter.process(trajectory)
