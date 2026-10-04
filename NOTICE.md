# Provenance and limits

Author implementation: jmpr0/incremental-nids, fixed commit 842737730852c7c7c0955257c48ca72a22770343. Its original license is reproduced unchanged in AUTHOR_LICENSE.

prepare_smoke.py creates a separate compatibility copy; setup_research.py creates the index-corrected research copy. The two diff files document these transformations. They are engineering corrections, not new learning algorithms.

run_research.py uses the original CNN and incremental training framework, adds GPU-compatible evaluation and predeclared replay/weighting ablations, and records fixed-sample limitations. Balanced replay and class weighting are established techniques; no first-in-the-world novelty is claimed.

Data remain governed by their original publishers' terms. A code license does not replace data permissions or citation requirements. For TON_IoT, follow the official eight-source attribution requirement: https://research.unsw.edu.au/projects/toniot-datasets . For IoT-NID, cite the dataset DOI 10.21227/q70p-q449.

Code preparation was assisted by an AI coding assistant. The project author remains responsible for reviewing the code, results, academic claims and attribution.
