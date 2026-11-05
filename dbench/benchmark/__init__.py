"""Controlled real-substrate benchmark of hypothesis-search methods (paper claim 1).

Planted relations of known class and effect size are added to the level-k
residual of a real kidney morphology target in the v0.5 measurement bank; every
search method sees the same proposal patients and the same hypothesis budget;
recovery, rank of the truth, relation class, held-out gain and wall-clock are
scored against the planted truth.

Modules: ``substrate`` (real or synthetic search design + comparison ladder),
``tasks`` (planted worlds, exact variance-fraction calibration), ``methods``
(the search methods behind one interface), ``scoring`` (recovery and held-out
gain), ``run`` (CLI: ``python -m dbench.benchmark.run --smoke``).
"""
