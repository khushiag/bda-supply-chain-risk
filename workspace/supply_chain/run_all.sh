#!/bin/bash
# Runs the whole pipeline on the YARN cluster. Use inside the master container:
#   bash /workspace/supply_chain/run_all.sh          # every stage
#   bash /workspace/supply_chain/run_all.sh 04 05    # only some stages
set -eo pipefail
cd "$(dirname "$0")"
LOGS=/workspace/output/logs; mkdir -p "$LOGS"
SEED_LOCAL=/workspace/data/DataCoSupplyChainDataset.csv

SUBMIT="spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-cores 2 --executor-memory 1g --driver-memory 1g \
  --conf spark.yarn.maxAppAttempts=1"

STAGES=${@:-01 02 03 04 05 06}

if [[ " $STAGES " == *" 01 "* ]]; then
  if ! hdfs dfs -test -e /dataco/seed/DataCoSupplyChainDataset.csv; then
    [ -f "$SEED_LOCAL" ] || { echo "Put DataCoSupplyChainDataset.csv in workspace/data/ first"; exit 1; }
    echo ">> Uploading seed CSV to HDFS"
    hdfs dfs -mkdir -p /dataco/seed
    hdfs dfs -put "$SEED_LOCAL" /dataco/seed/
  fi
fi

for s in $STAGES; do
  script=$(ls ${s}_*.py)
  echo "================ $script ================"
  start=$(date +%s)
  if [ "$s" = "06" ]; then
    python3 "$script" 2>&1 | tee "$LOGS/$script.log"
  else
    $SUBMIT "$script" 2>&1 | grep -v -E "^\s*$|INFO|WARN" | tee "$LOGS/$script.log"
  fi
  echo ">> $script finished in $(( $(date +%s) - start ))s"
done
echo "All done. Results: workspace/output/  (figures in workspace/output/figures)"
