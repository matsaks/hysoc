"""
Default parameters for behavioral STOP/MOVE segmentation (Module I).

These defaults are shared by STEP (online, engines/step.py) and STSS
(offline oracle, engines/stss_sklearn.py) so that the two implementations
operate on the same distance and duration thresholds. This is required for
a fair RQ1 comparison per the TDT4501 preproject, Sect. 4.6.1.

Sources
-------
STEP: Sun et al., "Streaming Trajectory Segmentation Based on Stay-Point
Detection", DASFAA 2024. Swept D in {10, 20, 50, 75, 100} m and
T in {1, 3, 5, 10, 15} s on CQ-Taxi / CQ-Hazs / Geolife.

STSS: Liu et al., "A Semantics-Based Trajectory Segmentation Simplification
Method", J. Geovis. Spat. Anal. 2021. Recommended MinPts = residence_time /
sampling_interval; used MinPts = 12-60 in their experiments.

Empirical tuning on NYC 1 Hz WorldTrace subset
----------------------------------------------
Sweep: eps in {10, 15, 20, 30, 50} m x T in {10, 15, 30, 60} s across
100 trajectories (see scripts/demo_27_step_stss_param_sweep.py and
data/processed/param_sweep_step_stss.csv).

Selection heuristic: smallest (eps, T) pair where median STEP yields
>= 2 stops and >= 2 moves and STEP-vs-STSS temporal-IoU F1 > 0.6.

Selected operating point: (eps=15 m, T=30 s), a balanced choice that
preserves high agreement (median F1 ≈ 0.76, matched IoU median ≈ 0.93)
while yielding slightly richer stop/move segmentation than eps=10 m.
"""

from __future__ import annotations

# Distance threshold for stop/stay-point neighborhood checks (meters).
# STEP: symbol D in the DASFAA paper. STSS: symbol epsilon in Liu et al.
# Selected 15 m as the best balance point for the thesis objective:
# keep high STEP-vs-STSS agreement while preserving richer stop/move counts
# for downstream HYSOC buffer-clearing events.
STOP_MAX_EPS_METERS: float = 15.0
STSS_MAX_EPS_METERS: float = 15.0

# Minimum duration for a dwell to be treated as a Stop (seconds).
# STEP: symbol T in the DASFAA paper. STSS: in the paper this is baked
# into MinPts; we surface it as an explicit knob so the two algorithms
# share an identical duration threshold for RQ1 comparability.
# Selected 30 s: the sweep shows that T <= 15 s cannot separate real dwells
# from traffic pauses on NYC 1 Hz (F1 < 0.5), while T = 30 s yields F1 = 0.80.
STOP_MIN_DURATION_SECONDS: float = 30.0
STSS_MIN_DURATION_SECONDS: float = 30.0

# Density parameter used by STSS' OPTICS clustering to form candidate stop
# clusters. Derived at 1 Hz as max(5, round(T * 0.5)): a 50 percent
# GPS-drop tolerance during the dwell window, bounded below by 5 to avoid
# degenerate clusters. For T = 30 s at 1 Hz, this gives 15.
STSS_MIN_SAMPLES: int = 15
