"""
Objective 4 - Predict late delivery AT THE MOMENT AN ORDER IS PLACED with Spark MLlib.

Models: Logistic Regression, Random Forest, Gradient-Boosted Trees.
Target: late_delivery_risk (1 = arrives after the promised date). Cancelled orders excluded.

No-leakage rule: only fields known when the order is placed are used. Excluded on purpose:
  days_real, shipping_ts, delivery_status, delay_days  -> they ARE the outcome
  order_status, order_profit_per_order, benefit, sales_per_customer -> settled after shipping
Split: by source_order_id (all copies of one seed order land on the same side), so the test set
never contains a near-duplicate of a training order. 70% train / 10% validation / 20% test.
The decision threshold for each model is tuned on VALIDATION, then reported once on TEST.

Run:  spark-submit 04_train.py [--tree-fraction 1.0]
"""
import argparse
import json

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark import StorageLevel

from common import CURATED, MODELS, Timer, get_spark, out_path, save_json

p = argparse.ArgumentParser()
p.add_argument("--tree-fraction", type=float, default=1.0,
               help="fraction of the training set used for RF/GBT (lower it if training is too slow)")
p.add_argument("--models", default="lr,rf,gbt")
args, _ = p.parse_known_args()

spark = get_spark("04-mllib-late-delivery-models")

CAT = ["shipping_mode", "market", "order_region", "category_name", "customer_segment", "type",
       "department_name", "carrier_id", "supplier_id", "warehouse_id"]
NUM = ["days_scheduled", "order_month", "order_dow", "order_item_quantity", "order_item_product_price",
       "order_item_discount_rate", "sales", "carrier_reliability", "supplier_lead_time_days",
       "weather_severity", "port_congestion_index", "disruption_flag", "disruption_severity"]
LABEL = "late_delivery_risk"
LEAKY = ["days_real", "shipping_ts", "delivery_status", "delay_days", "order_status",
         "order_profit_per_order", "benefit_per_order", "sales_per_customer"]
print("Features used:", CAT + NUM)
print("Excluded (leakage):", LEAKY)

df = spark.read.parquet(CURATED).filter(~F.col("is_canceled")) \
    .select(*CAT, *NUM, F.col(LABEL).cast("double").alias("label"), "source_order_id") \
    .withColumn("split", F.pmod(F.xxhash64("source_order_id", F.lit("split")), F.lit(100)))
train = df.filter("split < 70").drop("split", "source_order_id").persist(StorageLevel.MEMORY_AND_DISK)
valid = df.filter("split >= 70 AND split < 80").drop("split", "source_order_id").persist(StorageLevel.MEMORY_AND_DISK)
test = df.filter("split >= 80").drop("split", "source_order_id").persist(StorageLevel.MEMORY_AND_DISK)
sizes = {"train": train.count(), "validation": valid.count(), "test": test.count()}
print("Split sizes:", sizes)

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CAT]
idx_cols = [c + "_idx" for c in CAT]

# Logistic regression needs one-hot + scaling; trees work directly on category indexes
lr_stages = indexers + [
    OneHotEncoder(inputCols=idx_cols, outputCols=[c + "_oh" for c in CAT], handleInvalid="keep"),
    VectorAssembler(inputCols=[c + "_oh" for c in CAT] + NUM, outputCol="raw_features"),
    StandardScaler(inputCol="raw_features", outputCol="features"),
    LogisticRegression(maxIter=60, regParam=0.001, elasticNetParam=0.0)]
tree_assembler = VectorAssembler(inputCols=idx_cols + NUM, outputCol="features")
MODEL_DEFS = {
    "lr": ("Logistic Regression", lr_stages, 1.0),
    "rf": ("Random Forest", indexers + [tree_assembler, RandomForestClassifier(
        numTrees=60, maxDepth=10, maxBins=64, subsamplingRate=0.7, featureSubsetStrategy="sqrt", seed=7)],
        args.tree_fraction),
    "gbt": ("Gradient-Boosted Trees", indexers + [tree_assembler, GBTClassifier(
        maxIter=60, maxDepth=6, maxBins=64, stepSize=0.15, subsamplingRate=0.8, seed=7)], args.tree_fraction),
}


def prob_bins(pred_df, bins=200):
    """Distributed: count positives/negatives per probability bin. Small result -> pandas."""
    return pred_df.withColumn("p", vector_to_array("probability")[1]) \
        .withColumn("bin", F.least(F.floor(F.col("p") * bins), F.lit(bins - 1)).cast("int")) \
        .groupBy("bin").agg(F.sum("label").alias("pos"), F.sum(1 - F.col("label")).alias("neg")) \
        .toPandas().set_index("bin").reindex(range(bins), fill_value=0).sort_index()


def metrics_at(b, threshold_bin):
    tp = b.pos[threshold_bin:].sum(); fp = b.neg[threshold_bin:].sum()
    fn = b.pos[:threshold_bin].sum(); tn = b.neg[:threshold_bin].sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "precision": prec, "recall": rec,
            "f1": f1, "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def roc_curve(b):
    rows = []
    P, N = b.pos.sum(), b.neg.sum()
    for t in range(len(b) + 1):
        rows.append({"threshold": t / len(b), "tpr": b.pos[t:].sum() / P, "fpr": b.neg[t:].sum() / N})
    return pd.DataFrame(rows)


auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
results, best = [], None
for key in [m.strip() for m in args.models.split(",")]:
    name, stages, frac = MODEL_DEFS[key]
    fit_df = train if frac >= 1.0 else train.sample(fraction=frac, seed=7)
    with Timer(f"Training {name}") as t:
        model = Pipeline(stages=stages).fit(fit_df)
    with Timer(f"Evaluating {name}"):
        vbins = prob_bins(model.transform(valid))
        best_bin = max(range(20, 181), key=lambda i: metrics_at(vbins, i)["f1"])   # threshold 0.10..0.90
        threshold = best_bin / 200
        test_pred = model.transform(test).persist(StorageLevel.MEMORY_AND_DISK)
        tbins = prob_bins(test_pred)
        m = metrics_at(tbins, best_bin)
        m_default = metrics_at(tbins, 100)
        auc = auc_eval.evaluate(test_pred)
        test_pred.unpersist()
    roc_curve(tbins).to_csv(out_path("ml", f"roc_{key}.csv"), index=False)

    res = {"key": key, "model": name, "train_rows_used": fit_df.count() if frac < 1.0 else sizes["train"],
           "train_seconds": t.seconds, "threshold": threshold, "auc": auc,
           **{f"test_{k}": v for k, v in m.items()},
           "test_f1_at_0.5": m_default["f1"], "valid_f1": metrics_at(vbins, best_bin)["f1"]}
    clf = model.stages[-1]
    if hasattr(clf, "featureImportances"):
        imp = clf.featureImportances.toArray()
        names = [c.replace("_idx", "") for c in idx_cols] + NUM
        pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False) \
          .to_csv(out_path("ml", f"feature_importance_{key}.csv"), index=False)
    results.append(res)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))

    if best is None or res["valid_f1"] > best[0]["valid_f1"]:   # choose by VALIDATION, not test
        best = (res, model)

pd.DataFrame(results).to_csv(out_path("ml", "model_comparison.csv"), index=False)
best_res, best_model = best
with Timer(f"Saving best model ({best_res['model']}) to HDFS"):
    best_model.write().overwrite().save(f"{MODELS}/best_model")
save_json({"best_model": best_res["model"], "key": best_res["key"], "threshold": best_res["threshold"],
           "test_f1": best_res["test_f1"], "target_f1": 0.80, "target_met": best_res["test_f1"] >= 0.80,
           "split_sizes": sizes, "features": CAT + NUM, "excluded_leaky": LEAKY,
           "model_path": f"{MODELS}/best_model"}, "ml", "best_model.json")
print(f"\nBEST: {best_res['model']}  test F1 = {best_res['test_f1']:.4f}  (target 0.80)")
spark.stop()
