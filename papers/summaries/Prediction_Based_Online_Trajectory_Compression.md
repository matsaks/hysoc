# Summary: Prediction-based Online Trajectory Compression

**Key Findings**
* Existing trajectory compression techniques are designed for offline or batch settings and fail to address the online setting, where a compressed trajectory database must be maintained as new GPS updates arrive continuously.
* The paper introduces ONTRAC (ONline TRAjectory Compression), a framework for map-matched online trajectory compression that uses prediction models to suppress redundant database updates.
* ONTRAC outperforms PRESS — an offline baseline extended to an online setting with delay windows — achieving up to 13.6× (SF) and 10.4× (Beijing) higher spatial compression ratios and up to 21× higher temporal compression ratios, even when the baseline is allowed 4-minute update delays.

**Methodology and Main Contributions**
* Trajectories are decomposed into two independent components: a spatial component (sequence of road segment IDs) and a temporal component (travel-times per segment). Separate prediction models are trained and applied to each.
* Spatial compression uses a k-order Markov model (Ψ) trained on historical trajectories to predict the next road segment given the k most recent segments. A database update is suppressed whenever the prediction is correct; only segments that deviate from the prediction are stored. The result is spatially lossless compression.
* Temporal compression learns per-segment travel-time distributions via a Gaussian model whose parameters are estimated using an Expectation-Maximisation (EM) algorithm with a convex Quadratic Programming (QP) formulation. Predicted travel-times are fused with sparse GPS updates; a database update is suppressed when the fused estimate lies within a user-defined error bound λ. The result is temporally lossy with guaranteed error bound.
* A novel concept of spatial update block entropy (h_k) is defined to characterise the compressibility of a road network; it is shown to equal 1 − Σ π(s_i)/deg°(s_i), linking network topology (via PageRank) directly to compression hardness.
* Partial trajectory decompression (ONTRAC-PT) is introduced for efficient query answering: rather than reconstructing the full trajectory, a minimal context is traced backwards from the query time to achieve unique reconstruction, incurring only 38% (SF) and 25% (Beijing) overhead compared to no compression.

**Results**
* Evaluated on two real-world taxi datasets: San Francisco Cab (500 cabs, 30 days, 7.7M GPS updates) and Beijing (10,357 cabs, one week, 904K updates), plus two synthetic datasets.
* ONTRAC increases database insert throughput by factors of 11.0 (SF) and 11.7 (Beijing) over the no-compression baseline, and by up to one order of magnitude over PRESS.
* Spatial compression runs in O(|T|k) time; temporal compression in O(|T|) per trajectory; training converges in 5 EM iterations in experiments.
* Training on 12 million trajectories with 8 cores takes approximately 3 hours; online compression time is in the order of seconds.

**Limitations and Relevance**
* The Markov model for spatial compression requires sufficient historical data per road segment; segments unseen during training cannot be predicted.
* Temporal compression is error-bounded but lossy, and the Gaussian travel-time assumption may not hold for highly irregular mobility (e.g., pedestrian or multi-modal trajectories).
* No semantic stop/move segmentation is performed; the framework treats all movement uniformly as road-network traversal.
* Relevant to HYSOC as a contrasting online network-based approach: ONTRAC suppresses updates via prediction without explicit behavioural segmentation, whereas HYSOC-N explicitly segments STOPs and MOVEs before applying network encoding. ONTRAC's k-mer spatial compression directly informs the referential encoding strategy in TRACE and HYSOC-N.
