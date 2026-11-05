"""Learned scientific search: a dataset-conditioned model that selects small
variable sets (molecular, morphology concept, spatial-context transforms) worth
testing, with the relation shape left free.

Modules: ``episodes``, ``tokens``, ``model``, ``train``, ``ablation`` (the
model and its training) and ``propose`` (the bridge from a trained model to
frozen :class:`morpheus_encoder.general_discovery.Candidate` objects plus the
transformed design columns they use).  Import from the submodules, e.g.
``from morpheus_encoder.learned_search.propose import propose`` (the package
itself imports nothing, so it never pulls in torch).
"""
