"""edge_lab: new experimental pipeline for the brain-age task.

Key new idea vs the main project: a *structural* signal block of ~229 features
(largest connected component of the |Spearman corr| > 0.35 graph, computed
transductively on train+test -- unsupervised, no y, perfectly stable) that
contains 100% of the features with |rho_y| > 0.15. Feature selection is then
done *within* this block, so no noise feature can dilute the ranking.
"""
from .pipeline import (  # noqa: F401
    ROOT, DATA_DIR, SEED, load_data, compute_block_mask,
    SignedLog, LogMADMasker, BlockRestrictedSelector, PairAverager,
    ReliabilityWeighter, make_preprocessor, build_model, MODELS,
)
