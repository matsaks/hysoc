# Summary: Compression of Trajectory Data: A Comprehensive Evaluation and New Approach

**Key Findings**
* GPS trajectory data from mobile devices is growing exponentially, presenting major problems for data transmission, computational query costs, and disk storage.
* While existing compression algorithms can reduce the size of trajectory data, it is critical that they minimize the loss of spatial and temporal information essential to location-based applications.
* The proposed SQUISH-E (Spatial QUalIty Simplification Heuristic Extended) algorithm provides a flexible, fast, and accurate online/offline method for trajectory compression, outperforming or matching the accuracy of existing offline algorithms while achieving much faster execution times.

**Methodology and Main Contributions**
* SQUISH-E is an enhancement of the previous SQUISH algorithm, adding the crucial ability to compress trajectories with provable, mathematically guaranteed upper bounds on Synchronized Euclidean Distance (SED) error.
* The algorithm uses a dynamically sized priority queue. Each point is evaluated, and its priority is defined as an estimated upper bound of the SED error that its removal would cause. SQUISH-E selectively removes the points with the lowest priority to limit error growth. * It offers two specialized execution modes: SQUISH-E($\lambda$) minimizes SED error while strictly maintaining a user-defined compression ratio $\lambda$, and SQUISH-E($\mu$) maximizes the compression ratio while ensuring the SED error stays underneath a user-specified threshold $\mu$.
* The authors also executed a comprehensive empirical comparison of seven algorithms (Uniform Sampling, Douglas-Peucker, TD-TR, Opening Window, OPW-TR, Dead Reckoning, and SQUISH-E) against three real-world datasets spanning different transportation profiles (bus, urban commuter, and multi-modal travel). 
**Results**
* In tests measuring execution time, SQUISH-E proved to be 4 to 6 times faster than TD-TR (the other top-performing SED-based algorithm) because it only requires updating the priority of a maximum of two neighboring points when a point is removed from the queue.
* TD-TR and SQUISH-E($\mu$) produced the most accurate compressed trajectories overall regarding SED and speed error. Douglas-Peucker and Opening Window performed poorly in terms of SED and speed errors because they prioritize spatial error and ignore temporal data.
* The type of dataset significantly impacts compression accuracy. Trajectories from the complex, multi-modal GeoLife dataset were the hardest to compress accurately due to frequent, erratic changes in speed and heading, whereas smooth urban commuter data (NYMTC) was much easier to compress.

**Limitations and Future Work**
* While the paper establishes SQUISH-E as highly efficient, future work aims to incorporate road network knowledge into the compression logic. This could enable the rapid detection of deviations from the road network and support even more aggressive data compression.
* The authors also plan to test SQUISH-E's compression impact on actual spatial applications, such as real-time traffic flow modeling, bottleneck detection, and the automated identification of speeding hot-spots.