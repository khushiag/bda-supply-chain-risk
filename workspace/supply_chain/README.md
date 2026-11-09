# Supply Chain Risk Detection & Delivery Optimization (BDA DSE 3125)

Spark SQL + MLlib pipeline on the 3-node Hadoop/YARN cluster (`master`, `worker1`, `worker2`).

| Stage | Script | Synopsis objective | Output |
|---|---|---|---|
| 01 | `01_generate.py` | Obj 1: scale DataCo to ~3M lines, check within 2% | HDFS `/dataco/raw/*` (5 CSV tables), `output/validation/` |
| 02 | `02_etl.py` | Obj 3: clean, join 5 tables, Parquet | HDFS `/dataco/curated/order_lines` |
| 03 | `03_analysis.py` | Obj 3: 13 Spark SQL risk queries | `output/analysis/*.csv` |
| 04 | `04_train.py` | Obj 4: LR / RF / GBT, F1 ≥ 0.80 | `output/ml/`, HDFS `/dataco/models/best_model` |
| 05 | `05_optimize.py` | Obj 5: riskiest segments + what-if | `output/optimize/` |
| 06 | `06_visualize.py` | Obj 5: charts | `output/figures/*.png` |

## Run it
1. Download the seed from Mendeley (https://data.mendeley.com/datasets/8gx2fvg2k6/5) and put
   `DataCoSupplyChainDataset.csv` in `hadoop-docker/workspace/data/`.
2. Rebuild once (adds matplotlib, ~1 min, Hadoop/Spark stay cached): `docker compose up -d --build`
3. `docker exec -it master bash`, then:
   ```bash
   bash /workspace/supply_chain/run_all.sh          # all stages
   bash /workspace/supply_chain/run_all.sh 04 05 06 # re-run some stages
   ```
4. Watch jobs at http://localhost:8088 (YARN) and http://localhost:4040 (Spark) — screenshot them.

Knobs: `01_generate.py --target-rows 3500000` (bigger data), `--noise 0.40` (higher = harder to
predict); `04_train.py --tree-fraction 0.5` if RF/GBT training is too slow.

## Design decisions to explain in the viva
- **Stratified bootstrap at order level**: each seed order is copied 16 or 17 times, so every
  market × shipping-mode stratum keeps its share; copies get new IDs and keep `source_order_id`.
- **How the label is generated**: a risk score = weather + port congestion + carrier unreliability
  + supplier lead time + disruption severity + (weather×congestion interaction) + random noise.
  Within each shipping-mode × market stratum the riskiest X% are late, X = the seed's late rate for
  that stratum. This keeps late rates equal to the seed and gives the new fields a real signal.
  **Be upfront about this**: on the original DataCo columns alone (no leakage), late delivery is
  hard to predict (F1 ≈ 0.71, roughly "predict everything late"). The 0.80 target is reachable
  because our synthetic drivers carry signal by design; `--noise` controls how much.
- **No leakage**: `days_real`, `delivery_status`, shipping date, order status and profit are
  excluded — they are only known after delivery.
- **Group split**: train/validation/test is split by `source_order_id`, so copies of the same seed
  order never sit on both sides. Threshold tuned on validation, reported once on test.
- **Optimisation caveat**: switching First Class → Standard lowers "late" mostly because the
  promise becomes longer (1 → 4 days) — see `promise_change_days`. The +1 day buffer chart shows
  the same insight: much of the lateness is unrealistic promises, not slow shipping.
- **Shipping-cost assumption** (`05_optimize.py`): Standard 0%, Second +1.5%, First +3%,
  Same Day +6% of order value. State it in the report.
