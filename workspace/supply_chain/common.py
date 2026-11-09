"""Shared paths, Spark session and helpers for the supply-chain pipeline."""
import json
import os
import time

from pyspark.sql import SparkSession, functions as F

# Where data lives in HDFS, and where small results (CSVs, charts) go on the shared /workspace folder
BASE = os.environ.get("BDA_HDFS", "hdfs://master:9000/dataco")
OUT = os.environ.get("BDA_OUT", "/workspace/output")

SEED_CSV = f"{BASE}/seed/DataCoSupplyChainDataset.csv"
RAW = f"{BASE}/raw"
CURATED = f"{BASE}/curated/order_lines"
MODELS = f"{BASE}/models"

# Promised delivery days per shipping mode in DataCo
SCHEDULED_DAYS = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}


def get_spark(app_name):
    builder = SparkSession.builder.appName(app_name) \
        .config("spark.sql.shuffle.partitions", os.environ.get("BDA_SHUFFLE", "48")) \
        .config("spark.sql.adaptive.enabled", "true")
    if os.environ.get("BDA_LOCAL"):          # only for testing without the cluster
        builder = builder.master("local[2]").config("spark.driver.memory", "3g")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def out_path(*parts):
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_small(df, *parts):
    """Collect a SMALL Spark result to the driver and save it as CSV under /workspace/output."""
    pdf = df.toPandas()
    pdf.to_csv(out_path(*parts), index=False)
    return pdf


def save_json(obj, *parts):
    with open(out_path(*parts), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def uniform(col, salt):
    """Deterministic pseudo-random number in [0,1) from a column value and a salt.
    Using a hash (instead of rand()) keeps every line of the same order consistent."""
    h = F.xxhash64(col, F.lit(salt))
    return (F.pmod(h, F.lit(1_000_003)).cast("double")) / F.lit(1_000_003.0)


def normal(col, salt):
    """Standard-normal noise via Box-Muller on two hashed uniforms."""
    u1 = F.greatest(uniform(col, salt + "_a"), F.lit(1e-9))
    u2 = uniform(col, salt + "_b")
    return F.sqrt(-2.0 * F.log(u1)) * F.cos(2.0 * 3.141592653589793 * u2)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"\n>> {self.label} ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t, 1)
        print(f">> {self.label} done in {self.seconds}s", flush=True)
