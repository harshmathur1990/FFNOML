#!/bin/bash
# Learning curve on Olivia: does the FFNO gain more from more simulations or from more of the same one?
#
# Every experiment trains from scratch with the production model and settings (config.py) and is scored against MULTI3D on
# en024048_hion, which none of them sees, and on ch024031_by200bz005/450, which no checkpoint has seen:
#   sims1..sims4  one training snapshot from each of 1, 2, 3, 4 simulations (nested)
#   same3         three snapshots of one simulation: the volume of sims3 without its variety
#   all5          the production splits from scratch, for reference (sees en024048_hion; omit with EXPERIMENTS=...)
# If the held-out error falls from sims1 to sims4, and sims3 beats same3, diverse data is the lever.
#
# Run from the repository on an Olivia login node:
#   scripts/submit_learning_curve.sh [ROOT]
# ROOT (default /cluster/projects/nn2834k/harshm/learning_curve) gets one run directory per experiment, each with
# experiment.env, eval_snapshots.txt, the datasets, the checkpoint and eval/population_errors_<name>.json.
# Environment: EXPERIMENTS (space separated, default all six), ATOMS (H, the default, or CA: one per ROOT), EPOCHS (default 400), NODES (default 2),
# DATA_DIR (default /cluster/projects/nn2834k/harshm/, holding bifrost_data/), DRY_RUN=1 to write the run directories only.
# A training job stopped by its time limit continues with TRAIN_FLAGS=--resume EXPERIMENTS=<name> scripts/submit_learning_curve.sh ROOT
# (its datasets are kept and the build is skipped).
# Afterwards, on eagle7 or anywhere: python scripts/learning_curve_summary.py ROOT

set -o errexit
set -o nounset
set -o pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${1:-/cluster/projects/nn2834k/harshm/learning_curve}
DATA_DIR=${DATA_DIR:-/cluster/projects/nn2834k/harshm/}
ATOMS=${ATOMS:-H}
EPOCHS=${EPOCHS:-400}
NODES=${NODES:-2}
EXPERIMENTS=${EXPERIMENTS:-"sims1 sims2 sims3 sims4 same3 all5"}
[[ "${ATOMS}" != *,* ]] || { echo "ATOMS=${ATOMS}: one atom per root, as production trains H and CA apart (run again with ATOMS=CA and another ROOT)"; exit 1; }

# name | training split | validation split  (JSON, {"sim": ["snap", ...]})
declare -A TRAIN VAL
TRAIN[sims1]='{"nw012023": ["915"]}'
VAL[sims1]='{"nw012023": ["940"]}'
TRAIN[sims2]='{"nw012023": ["915"], "ch012006": ["836"]}'
VAL[sims2]='{"nw012023": ["940"], "ch012006": ["849"]}'
TRAIN[sims3]='{"nw012023": ["915"], "ch012006": ["836"], "ch012012_hion": ["910"]}'
VAL[sims3]='{"nw012023": ["940"], "ch012006": ["849"], "ch012012_hion": ["984"]}'
TRAIN[sims4]='{"nw012023": ["915"], "ch012006": ["836"], "ch012012_hion": ["910"], "qs006003_sap": ["689"]}'
VAL[sims4]='{"nw012023": ["940"], "ch012006": ["849"], "ch012012_hion": ["984"], "qs006003_sap": ["900"]}'
TRAIN[same3]='{"nw012023": ["915", "1050", "1120"]}'
VAL[same3]='{"nw012023": ["940"]}'
TRAIN[all5]='{"en024048_hion": ["465"], "nw012023": ["915"], "ch012012_hion": ["910"], "ch012006": ["836"], "qs006003_sap": ["689"]}'
VAL[all5]='{"en024048_hion": ["700"], "nw012023": ["940"], "ch012012_hion": ["984"], "ch012006": ["849"], "qs006003_sap": ["900"]}'

# snapshots the experiments never train on
HELD_OUT="unseen_sim=en024048_hion/465
unseen_sim=en024048_hion/700
unseen_sim=en024048_hion/385
unseen_sim=ch024031_by200bz005/450"

snapshot_lines() {   # role, JSON split -> role=sim/snap lines
  python3 -c 'import json,sys; [print(f"{sys.argv[1]}={s}/{n}") for s,v in json.loads(sys.argv[2]).items() for n in v]' "$1" "$2"
}

# every snapshot must have its atmosphere and MULTI3D output, or the build fails hours into the queue
check_split() {
  python3 - "$DATA_DIR" "$ATOMS" "$1" <<'PY'
import json, os, sys
data, atoms, split = sys.argv[1], sys.argv[2].split(","), json.loads(sys.argv[3])
missing = [p for s, snaps in split.items() for n in snaps
           for p in [f"{data}/bifrost_data/{s}/{n}/{f}" for f in ("atm3d", "mesh")] + [f"{data}/bifrost_data/{s}/{n}/{a}/out_pop" for a in atoms]
           if not os.path.exists(p)]
if missing:
    sys.exit("missing:\n  " + "\n  ".join(missing))
PY
}

mkdir -p "${ROOT}/solving_sets"
previous_eval=""
for name in ${EXPERIMENTS}; do
  [[ -n "${TRAIN[$name]:-}" ]] || { echo "unknown experiment ${name}"; exit 1; }
  check_split "${TRAIN[$name]}"; check_split "${VAL[$name]}"
  run="${ROOT}/${name}"; mkdir -p "${run}"
  cat > "${run}/experiment.env" <<EOF
FFNOML_EXPERIMENT='${name}'
FFNOML_REPO_DIR='${REPO}'
FFNOML_DATA_DIR='${DATA_DIR}'
FFNOML_TRAIN_SPLIT='${TRAIN[$name]}'
FFNOML_VAL_SPLIT='${VAL[$name]}'
FFNOML_ACTIVE_ATOMS='${ATOMS}'
FFNOML_NUM_EPOCHS='${EPOCHS}'
FFNOML_TRAIN_FLAGS='${TRAIN_FLAGS:-}'
EOF
  { snapshot_lines train "${TRAIN[$name]}"; snapshot_lines val "${VAL[$name]}"
    if [[ "${name}" == all5 ]]; then   # it trains on en024048_hion: 385 is a further snapshot of a seen simulation
      echo "same_sim=en024048_hion/385"; echo "${HELD_OUT}" | grep ch024031
    else
      echo "${HELD_OUT}"
    fi; } > "${run}/eval_snapshots.txt"
  echo "${name}: $(wc -l < "${run}/eval_snapshots.txt") evaluation snapshots in ${run}"
  [[ "${DRY_RUN:-0}" == 1 ]] && continue
  train_id=$(sbatch --parsable --nodes="${NODES}" --job-name="lc_${name}" --chdir="${run}" \
    --output="${run}/slurm-%j.out" --error="${run}/slurm-%j.err" \
    --export=ALL,FFNOML_RUN_DIR="${run}" "${REPO}/scripts/learning_curve_train.sbatch")
  dependency="afterok:${train_id}${previous_eval:+,afterany:${previous_eval}}"
  previous_eval=$(sbatch --parsable --job-name="lc_${name}_eval" --chdir="${run}" --dependency="${dependency}" \
    --output="${run}/slurm-eval-%j.out" --error="${run}/slurm-eval-%j.err" \
    --export=ALL,FFNOML_RUN_DIR="${run}" "${REPO}/scripts/learning_curve_eval.sbatch")
  echo "  training job ${train_id}, evaluation job ${previous_eval}"
done
