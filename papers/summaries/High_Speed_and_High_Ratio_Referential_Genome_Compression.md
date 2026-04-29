# Summary: High-speed and high-ratio referential genome compression

**Key Findings**
* Traditional text compression tools (gzip, bzip2) are inefficient for DNA sequences due to their small alphabet (A, C, G, T), frequent repeats, and palindromes; referential compression — storing only differences between a target genome and a reference genome — offers substantially higher compression ratios.
* The paper presents HiRGC, a referential genome compression algorithm based on 2-bit encoding and a greedy hash-table matching strategy, achieving compression ratios of 82–217× on eight benchmark human genomes.
* HiRGC outperforms four state-of-the-art methods (GDC-2, iDoComp, ERGC, NRGC) by at least 1.9× on compressed size and is at least 2.9× faster, completing compression of ~21 GB in under 30 minutes.

**Methodology and Main Contributions**
* Preprocessing converts a FASTA genome file to a 2-bit integer sequence (A=0, C=1, G=2, T=3), separating auxiliary information (identifiers, lowercase letters, N-positions) for independent encoding. This reduces the effective alphabet to Ψ = {A, C, G, T}, enabling all downstream operations on compact 2-bit integers.
* A global hash table is constructed from all k-tuple values of the reference sequence (k=20 recommended). Each k-tuple V^m_i = Σ u(i+j)×4^j is hashed to a bucket; buckets chain entries by next-pointer.
* Greedy matching scans the target sequence left-to-right: for each position, the k-tuple is looked up in the hash table, and the longest extension among all colliding reference positions is selected as the match. Unmatched substrings are recorded as literals (mismatched subsequences). This single-pass strategy contrasts with iterative or block-based competitors, giving O(1.7 × n_r) average-case complexity.
* Post-processing applies delta encoding for match positions, run-length encoding for short sequences, and PPMD (a variant of Prediction by Partial Matching) as a final entropy coder on the residual text file.
* Decompression is linear and fully lossless: the reference sequence plus match positions and mismatched subsequences recover the target exactly.

**Results**
* On 56 reference-target pairs from eight human genomes, HiRGC achieves an average compressed size of 18.3 MB versus 227.5 MB (GDC-2), 303.7 MB (ERGC), 279.3 MB (NRGC), and 83.3 MB (iDoComp). HiRGC won best compression on 35 of the 56 pairs.
* Compression time is under 7 min (C++) for most cases; GDC-2 typically takes over 30 min and iDoComp over 30 min to an hour. NRGC failed to compress several chromosomes under default parameters.
* Results on 100 human genomes from the 1000 Genomes Project and multi-species genomes (C. elegans, yeast, Arabidopsis, rice) confirm consistently excellent performance.
* Optimal tuple length k=20; performance degrades slightly below k=10 and above k=25.

**Limitations and Relevance**
* Requires a reference genome to be provided; compression quality depends on sequence similarity between reference and target (>99% for human genomes). Not applicable when no suitable reference exists.
* The method is domain-specific to genomic FASTA sequences; direct transfer to GPS trajectory streams is not possible.
* Relevant to HYSOC indirectly through TRACE: TRACE's multi-reference k-mer matching for road-network trajectory compression adapts the same conceptual primitive — building a hash structure over reference k-tuples and greedy-matching a target stream against it — to the trajectory domain. HiRGC demonstrates the effectiveness of single-pass greedy hash matching as the core of a referential compression pipeline, which motivates TRACE's and HYSOC-N's referential encoding design.
