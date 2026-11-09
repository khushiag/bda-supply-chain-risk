"""
Objective 1 - Scale the DataCo seed (~180k order lines) into DataCo-Scaled (~3M lines, ~1.5 GB CSV).

How it works
  1. Read the seed CSV from HDFS and drop personal fields (names, emails, passwords, street).
  2. Stratified bootstrap: every seed ORDER is copied floor(k) or floor(k)+1 times (k = target/seed),
     so every (market x shipping mode) stratum keeps its original share.
  3. Add noise to prices (+-5%) and profit ratios, recompute sales/discount/profit consistently.
  4. Shift each copy by 0-7 whole years so order dates span 2015-2025 but seasonality is kept.
  5. Add ~10 new fields: carrier (+reliability), supplier (+lead time), warehouse, weather severity,
     port congestion, and disruption events (separate table, joined by region + date).
  6. Late-delivery label: a latent risk score is built from those drivers + random noise; inside each
     (shipping mode x market) stratum the riskiest X% of lines are marked late, where X% is that
     stratum's late rate in the seed. Late rates therefore match the seed, and the new fields carry a
     real, tunable signal (--noise: higher = weaker signal = harder prediction).
  7. Write 5 CSV tables to HDFS (orders, shipments, carriers, suppliers, disruptions) and a
     seed-vs-generated validation report (pass = within 2 percentage points / 2%).

Run:  spark-submit 01_generate.py [--target-rows 3000000] [--noise 0.40]
"""
import argparse
import random
from datetime import date, timedelta

from pyspark.sql import Window, functions as F

from common import (BASE, RAW, SEED_CSV, Timer, get_spark, normal, save_json, save_small, uniform)

p = argparse.ArgumentParser()
p.add_argument("--seed-csv", default=SEED_CSV)
p.add_argument("--target-rows", type=int, default=3_000_000)
p.add_argument("--noise", type=float, default=0.40, help="std-dev of random noise in the risk score")
p.add_argument("--random-seed", type=int, default=42)
args, _ = p.parse_known_args()

spark = get_spark("01-generate-dataco-scaled")
rng = random.Random(args.random_seed)
SALT = str(args.random_seed)

RENAME = {
    "Days for shipping (real)": "days_real",
    "Days for shipment (scheduled)": "days_scheduled",
    "order date (DateOrders)": "order_date_raw",
    "shipping date (DateOrders)": "shipping_date_raw",
    "Late_delivery_risk": "late_delivery_risk",
}
PII = ["customer_email", "customer_fname", "customer_lname", "customer_password",
       "customer_street", "customer_zipcode", "product_description"]


def snake(c):
    return RENAME.get(c) or "".join(ch.lower() if ch.isalnum() else "_" for ch in c).strip("_").replace("__", "_")


# ---------------------------------------------------------------- 1. seed
with Timer("Reading seed"):
    raw = spark.read.option("header", True).option("encoding", "ISO-8859-1").csv(args.seed_csv)
    seed = raw.toDF(*[snake(c) for c in raw.columns]).drop(*PII)
    ints = ["days_real", "days_scheduled", "late_delivery_risk", "category_id", "customer_id", "department_id",
            "order_customer_id", "order_id", "order_item_cardprod_id", "order_item_id", "order_item_quantity",
            "product_card_id", "product_category_id", "product_status"]
    dbls = ["benefit_per_order", "sales_per_customer", "latitude", "longitude", "order_item_discount",
            "order_item_discount_rate", "order_item_product_price", "order_item_profit_ratio", "sales",
            "order_item_total", "order_profit_per_order", "product_price"]
    for c in ints:
        seed = seed.withColumn(c, F.col(c).cast("int"))
    for c in dbls:
        seed = seed.withColumn(c, F.col(c).cast("double"))
    seed = seed.withColumn("order_ts", F.to_timestamp("order_date_raw", "M/d/yyyy H:mm")) \
               .withColumn("order_region", F.regexp_replace(F.trim("order_region"), r"\s+", " ")) \
               .drop("order_date_raw", "shipping_date_raw", "order_customer_id") \
               .filter(F.col("order_item_id").isNotNull() & F.col("order_ts").isNotNull())
    seed = seed.withColumn("is_canceled", F.col("delivery_status") == "Shipping canceled").cache()
    n_seed = seed.count()
    print(f"   seed rows: {n_seed:,}")

k = args.target_rows / n_seed
base_copies, frac = int(k), k - int(k)
print(f"   scale factor k = {k:.3f}  (each order copied {base_copies} or {base_copies + 1} times)")

# ---------------------------------------------------------------- 2. stratified bootstrap
oid = F.col("order_id")
gen = seed.withColumn("n_copies", F.lit(base_copies) + (uniform(oid, SALT + "rep") < frac).cast("int")) \
          .filter("n_copies > 0") \
          .withColumn("replica", F.explode(F.sequence(F.lit(0), F.col("n_copies") - 1))) \
          .withColumn("source_order_id", oid) \
          .withColumn("order_id", oid * 32 + F.col("replica")) \
          .withColumn("order_item_id", F.col("order_item_id") * 32 + F.col("replica"))
noid, item = F.col("order_id"), F.col("order_item_id")

# ---------------------------------------------------------------- 3. numeric noise (kept consistent)
gen = gen.withColumn("order_item_product_price",
                     F.round(F.greatest(F.lit(0.5), F.col("order_item_product_price") * (1 + 0.05 * normal(item, SALT + "p"))), 2)) \
    .withColumn("product_price", F.col("order_item_product_price")) \
    .withColumn("sales", F.round(F.col("order_item_product_price") * F.col("order_item_quantity"), 2)) \
    .withColumn("order_item_discount", F.round(F.col("sales") * F.col("order_item_discount_rate"), 2)) \
    .withColumn("order_item_total", F.round(F.col("sales") - F.col("order_item_discount"), 2)) \
    .withColumn("order_item_profit_ratio",
                F.round(F.least(F.lit(0.5), F.greatest(F.lit(-2.75), F.col("order_item_profit_ratio") + 0.02 * normal(item, SALT + "r"))), 2)) \
    .withColumn("order_profit_per_order", F.round(F.col("order_item_total") * F.col("order_item_profit_ratio"), 2)) \
    .withColumn("benefit_per_order", F.col("order_profit_per_order")) \
    .withColumn("sales_per_customer", F.col("order_item_total"))

# ---------------------------------------------------------------- 4. spread dates over 2015-2025
gen = gen.withColumn("year_shift", F.floor(uniform(noid, SALT + "yr") * 8).cast("int")) \
         .withColumn("order_ts", F.expr("order_ts + make_interval(year_shift)")) \
         .withColumn("order_month", F.month("order_ts")).withColumn("order_year", F.year("order_ts"))

# ---------------------------------------------------------------- 5. new fields
MARKETS = ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"]
market_idx = F.coalesce(*[F.when(F.col("market") == m, F.lit(i)) for i, m in enumerate(MARKETS)], F.lit(0))

# 15 carriers: 3 per market, each with an on-time reliability
carriers = []
for i, m in enumerate(MARKETS):
    for j in range(3):
        cid = i * 3 + j + 1
        carriers.append((f"CR{cid:02d}", f"{m.split()[0]} Freight {chr(65 + j)}", m,
                         round(rng.uniform(0.82, 0.97), 3)))
carriers_df = spark.createDataFrame(carriers, "carrier_id string, carrier_name string, home_market string, carrier_reliability double")

# 40 suppliers with a base lead time
countries = ["China", "India", "Vietnam", "Mexico", "Germany", "USA", "Brazil", "Turkey", "Bangladesh", "Poland"]
suppliers = [(f"SP{s:02d}", f"Supplier {s:02d}", rng.choice(countries), rng.randint(3, 24), round(rng.uniform(0.7, 0.99), 2))
             for s in range(1, 41)]
suppliers_df = spark.createDataFrame(suppliers, "supplier_id string, supplier_name string, supplier_country string, base_lead_time_days int, quality_score double")

gen = gen.withColumn("carrier_id", F.format_string("CR%02d", market_idx * 3 + F.pmod(F.xxhash64(noid, F.lit(SALT + "c")), F.lit(3)) + 1)) \
         .withColumn("supplier_id", F.format_string("SP%02d", F.pmod(F.col("category_id") * 3 + (uniform(item, SALT + "s") < 0.5).cast("int"), F.lit(40)) + 1)) \
         .withColumn("warehouse_id", F.format_string("WH-%s-%d", F.upper(F.substring("market", 1, 3)), F.pmod(F.xxhash64(noid, F.lit(SALT + "w")), F.lit(2)) + 1))

gen = gen.join(F.broadcast(carriers_df.select("carrier_id", "carrier_reliability")), "carrier_id") \
         .join(F.broadcast(suppliers_df.select("supplier_id", "base_lead_time_days")), "supplier_id") \
         .withColumn("supplier_lead_time_days",
                     F.round(F.greatest(F.lit(1.0), F.col("base_lead_time_days") * (1 + 0.25 * normal(item, SALT + "lt"))), 1)) \
         .drop("base_lead_time_days")

# Weather: seasonal peak month differs per market, with an occasional severe tail
peak = F.coalesce(*[F.when(F.col("market") == m, F.lit(pm)) for m, pm in
                    [("Pacific Asia", 8), ("Europe", 1), ("USCA", 1), ("LATAM", 2), ("Africa", 7)]], F.lit(1))
seasonal = 0.5 * (1 + F.cos(2 * 3.141592653589793 * (F.col("order_month") - peak) / 12))
gen = gen.withColumn("weather_severity", F.round(F.least(F.lit(1.0),
                     0.10 + 0.45 * seasonal * uniform(noid, SALT + "wx1") + 0.45 * F.pow(uniform(noid, SALT + "wx2"), 3)), 3))

# Port congestion: base per region + a 2020-2023 surge + noise
surge = F.when(F.col("order_year").isin(2021, 2022), 0.25).when(F.col("order_year").isin(2020, 2023), 0.10).otherwise(0.0)
gen = gen.withColumn("port_congestion_index", F.round(F.least(F.lit(1.0), F.greatest(F.lit(0.0),
                     0.15 + 0.35 * uniform(F.col("order_region"), SALT + "cg") + surge + 0.10 * normal(noid, SALT + "cg"))), 3))

# Disruption events table (strikes, storms, closures...) per region and date window
regions = [r.order_region for r in seed.select("order_region").distinct().collect()]
TYPES = [("Port strike", 2), ("Cyclone / storm", 3), ("Customs delay", 1), ("Port closure", 3),
         ("Labour shortage", 1), ("Lockdown", 3), ("Rail / road blockage", 2)]
events, start = [], date(2015, 1, 1)
for e in range(1, 361):
    etype, sev = rng.choice(TYPES)
    s = start + timedelta(days=rng.randint(0, 11 * 365))
    events.append((f"EV{e:04d}", rng.choice(regions), etype, sev, s, s + timedelta(days=rng.randint(3, 30))))
events_df = spark.createDataFrame(events, "event_id string, order_region string, event_type string, severity int, start_date date, end_date date")

orders_only = gen.select("order_id", "order_region", F.to_date("order_ts").alias("d")).dropDuplicates(["order_id"])
order_disr = orders_only.join(F.broadcast(events_df), on=[
    orders_only.order_region == events_df.order_region,
    orders_only.d.between(events_df.start_date, events_df.end_date)]) \
    .groupBy("order_id").agg(F.max("severity").alias("disruption_severity"))
gen = gen.join(order_disr, "order_id", "left").fillna({"disruption_severity": 0})

# ---------------------------------------------------------------- 6. late-delivery label
w_order = Window.partitionBy("order_id")
gen = gen.withColumn("order_lead", F.max("supplier_lead_time_days").over(w_order)) \
    .withColumn("risk_score",
                1.2 * F.col("weather_severity") + 1.0 * F.col("port_congestion_index")
                + 0.8 * (0.97 - F.col("carrier_reliability")) / 0.15
                + 0.6 * F.least(F.lit(1.0), F.col("order_lead") / 30)
                + 0.5 * F.col("disruption_severity")
                # non-linear effect: bad weather only really hurts when ports are already congested
                + 0.7 * ((F.col("weather_severity") > 0.55) & (F.col("port_congestion_index") > 0.5)).cast("int")
                + args.noise * normal(noid, SALT + "z"))

stratum_rate = seed.filter(~F.col("is_canceled")).groupBy("shipping_mode", "market") \
                   .agg(F.avg("late_delivery_risk").alias("seed_late_rate"))
active = gen.filter(~F.col("is_canceled")).join(F.broadcast(stratum_rate), ["shipping_mode", "market"]) \
    .withColumn("pr", F.percent_rank().over(Window.partitionBy("shipping_mode", "market").orderBy("risk_score")))
u_d1, u_d2 = uniform(noid, SALT + "d1"), uniform(noid, SALT + "d2")
active = active.withColumn("late_delivery_risk", (F.col("pr") >= 1 - F.col("seed_late_rate")).cast("int")) \
    .withColumn("days_real", F.when(F.col("late_delivery_risk") == 1,
                F.col("days_scheduled") + 1 + (u_d1 < 0.35).cast("int") + (u_d2 < 0.12).cast("int")
                + (F.col("disruption_severity") >= 2).cast("int"))
                .otherwise(F.greatest(F.lit(0), F.col("days_scheduled") - (u_d1 < 0.30).cast("int") - (u_d2 < 0.08).cast("int")))) \
    .withColumn("delivery_status", F.when(F.col("late_delivery_risk") == 1, "Late delivery")
                .when(F.col("days_real") < F.col("days_scheduled"), "Advance shipping").otherwise("Shipping on time")) \
    .drop("pr", "seed_late_rate")
canceled = gen.filter(F.col("is_canceled"))   # cancelled orders keep their seed status
gen = active.unionByName(canceled).withColumn("shipping_ts", F.expr("order_ts + make_interval(0,0,0,days_real)"))

# ---------------------------------------------------------------- 7. write the tables
ORDER_COLS = ["order_item_id", "order_id", "source_order_id", "order_ts", "customer_id", "customer_segment",
              "customer_city", "customer_state", "customer_country", "type", "market", "order_region",
              "order_country", "order_city", "order_state", "order_zipcode", "latitude", "longitude",
              "department_id", "department_name", "category_id", "category_name", "product_card_id",
              "product_category_id", "order_item_cardprod_id", "product_name", "product_image", "product_status",
              "product_price", "order_item_product_price", "order_item_quantity", "order_item_discount_rate",
              "order_item_discount", "sales", "order_item_total", "order_item_profit_ratio",
              "order_profit_per_order", "benefit_per_order", "sales_per_customer", "order_status"]
SHIP_COLS = ["order_item_id", "order_id", "shipping_mode", "days_scheduled", "days_real", "shipping_ts",
             "delivery_status", "late_delivery_risk", "carrier_id", "supplier_id", "warehouse_id",
             "supplier_lead_time_days", "weather_severity", "port_congestion_index"]

csv_opts = {"header": "true", "timestampFormat": "yyyy-MM-dd HH:mm:ss", "dateFormat": "yyyy-MM-dd"}
with Timer("Generating + writing orders/shipments (the heavy step)"):
    gen = gen.persist()          # reused by both table writes
    gen.select(*ORDER_COLS).write.mode("overwrite").options(**csv_opts).csv(f"{RAW}/orders")
    gen.select(*SHIP_COLS).write.mode("overwrite").options(**csv_opts).csv(f"{RAW}/shipments")
carriers_df.coalesce(1).write.mode("overwrite").options(**csv_opts).csv(f"{RAW}/carriers")
suppliers_df.coalesce(1).write.mode("overwrite").options(**csv_opts).csv(f"{RAW}/suppliers")
events_df.coalesce(1).write.mode("overwrite").options(**csv_opts).csv(f"{RAW}/disruptions")

# ---------------------------------------------------------------- 8. validation: seed vs generated
with Timer("Validating generated data against the seed"):
    def profile(df):
        n = df.count()
        r = {"rows": n,
             "late_rate": df.agg(F.avg("late_delivery_risk")).first()[0],
             "canceled_share": df.agg(F.avg(F.col("is_canceled").cast("int"))).first()[0],
             "mean_item_price": df.agg(F.avg("order_item_product_price")).first()[0],
             "mean_quantity": df.agg(F.avg("order_item_quantity")).first()[0]}
        for row in df.groupBy("shipping_mode").agg(F.count("*").alias("n"), F.avg("late_delivery_risk").alias("lr")).collect():
            r[f"mode_share::{row.shipping_mode}"] = row.n / n
            r[f"late_rate::{row.shipping_mode}"] = row.lr
        for row in df.groupBy("market").count().collect():
            r[f"market_share::{row.market}"] = row["count"] / n
        return r

    s, g = profile(seed), profile(gen)
    rows = []
    for key in s:
        if key == "rows":
            continue
        if key.startswith("mean_"):
            diff = (g[key] - s[key]) / s[key] * 100
            rows.append((key, round(s[key], 4), round(g[key], 4), round(diff, 2), "% (relative)", abs(diff) <= 2))
        else:
            diff = (g.get(key, 0) - s[key]) * 100
            rows.append((key, round(s[key], 4), round(g.get(key, 0), 4), round(diff, 2), "pp", abs(diff) <= 2))
    val = spark.createDataFrame(rows, "metric string, seed double, generated double, difference double, unit string, within_2 boolean")
    val.show(40, truncate=False)
    save_small(val, "validation", "seed_vs_generated.csv")

    mb = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(RAW)
    fs = mb.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    size_mb = fs.getContentSummary(mb).getLength() / 1024 / 1024
    summary = {"seed_rows": n_seed, "generated_rows": g["rows"], "scale_factor": round(k, 3),
               "raw_csv_size_mb": round(size_mb, 1), "noise": args.noise,
               "all_checks_within_2": bool(all(r[-1] for r in rows)), "hdfs_raw_path": RAW}
    save_json(summary, "validation", "generation_summary.json")
    print(summary)
spark.stop()
