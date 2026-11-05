# Sourced by every stage script. The code keeps its original box layout (see README, "Directory mapping").
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}" HDF5_USE_FILE_LOCKING=FALSE
PY="${PY:-python}"
WORKERS="${WORKERS:-6}"
STUDIES="gse237183_glioma nakajima_luad_b3 gse307534_lung_precursor_b3"   # untouched test studies (directory labels)
STUDIES_CSV="gse237183_glioma,nakajima_luad_b3,gse307534_lung_precursor_b3"
SUPPORT="meylan2022_ccrcc"
QUERIES="li2022_ccrcc gse250163 zenodo14620362"                          # no-refit transfer cohorts (B2 / MC1)
OURS="sel_screen,sel_screen_joint,random,linear_screen,nid"              # GPU stage methods
CPU_METHODS="ebm,gbm_shap,gbm_h,misty,stability_selection,rulefit,irf,knockoffs,pysr"
FAIR_METHODS="fair_gbm,fair_rulefit,fair_ebm,fair_irf,fair_misty"
ALL_METHODS="$OURS,$CPU_METHODS,$FAIR_METHODS"
cd "$PKG"
