"""
Objective 3a - Clean the raw CSV tables, join them and store an analysis-ready Parquet table.

  raw/orders + raw/shipments   (3M rows each)  --join on order_item_id-->
  + carriers, suppliers         (broadcast joins, small dimension tables)
  + disruptions                 (range join: same region AND order date inside the event window)
  --> curated/order_lines  (Parquet, partitioned by order_year)

Run:  spark-submit 02_etl.py
"""
from pyspark.sql import functions as F

from common import CURATED, RAW, Timer, get_spark, save_json

spark = get_spark("02-etl-clean-join-parquet")
read = lambda name: spark.read.option("header", True).option("inferSchema", False).csv(f"{RAW}/{name}")

with Timer("Reading raw tables"):
    orders, ships = read("orders"), read("shipments")
    carriers, suppliers, events = read("carriers"), read("suppliers"), read("disruptions")

# ------------------------------------------------------------- typing + cleaning
INT = ["order_item_id", "order_id", "source_order_id", "customer_id", "department_id", "category_id",
       "product_card_id", "order_item_quantity"]
DBL = ["latitude", "longitude", "product_price", "order_item_product_price", "order_item_discount_rate",
       "order_item_discount", "sales", "order_item_total", "order_item_profit_ratio", "order_profit_per_order"]
for c in INT:
    orders = orders.withColumn(c, F.col(c).cast("int"))
for c in DBL:
    orders = orders.withColumn(c, F.col(c).cast("double"))
orders = orders.withColumn("order_ts", F.to_timestamp("order_ts", "yyyy-MM-dd HH:mm:ss"))
for c in ["market", "order_region", "order_country", "category_name", "department_name", "customer_segment", "type"]:
    orders = orders.withColumn(c, F.regexp_replace(F.trim(c), r"\s+", " "))

ships = ships.select(
    F.col("order_item_id").cast("int"), "shipping_mode", "carrier_id", "supplier_id", "warehouse_id", "delivery_status",
    F.col("days_scheduled").cast("int"), F.col("days_real").cast("int"),
    F.to_timestamp("shipping_ts", "yyyy-MM-dd HH:mm:ss").alias("shipping_ts"),
    F.col("late_delivery_risk").cast("int"),
    F.col("supplier_lead_time_days").cast("double"), F.col("weather_severity").cast("double"),
    F.col("port_congestion_index").cast("double"))

with Timer("Cleaning"):
    n_before = orders.count()
    orders = orders.dropna(subset=["order_item_id", "order_id", "order_ts", "market"]) \
                   .filter((F.col("order_item_quantity") > 0) & (F.col("order_item_product_price") > 0)) \
                   .dropDuplicates(["order_item_id"])
    ships = ships.dropna(subset=["order_item_id", "shipping_mode", "days_scheduled"]).dropDuplicates(["order_item_id"])

carriers = carriers.select("carrier_id", "carrier_name", F.col("carrier_reliability").cast("double"))
suppliers = suppliers.select("supplier_id", "supplier_name", "supplier_country",
                             F.col("base_lead_time_days").cast("int"), F.col("quality_score").cast("double").alias("supplier_quality"))
events = events.select("event_id", F.col("order_region").alias("ev_region"), "event_type", F.col("severity").cast("int"),
                       F.to_date("start_date").alias("start_date"), F.to_date("end_date").alias("end_date"))

# ------------------------------------------------------------- joins
lines = orders.join(ships, "order_item_id", "inner") \
              .join(F.broadcast(carriers), "carrier_id", "left") \
              .join(F.broadcast(suppliers), "supplier_id", "left")

# Range join with disruption events, done once per ORDER (not per line) then joined back
order_keys = lines.select("order_id", "order_region", F.to_date("order_ts").alias("d")).dropDuplicates(["order_id"])
disr = order_keys.join(F.broadcast(events),
                       (order_keys.order_region == events.ev_region) & order_keys.d.between(events.start_date, events.end_date)) \
    .groupBy("order_id").agg(F.max("severity").alias("disruption_severity"),
                             F.first("event_type").alias("disruption_type"),
                             F.count("*").alias("active_events"))
lines = lines.join(disr, "order_id", "left") \
    .fillna({"disruption_severity": 0, "active_events": 0, "disruption_type": "None"}) \
    .withColumn("disruption_flag", (F.col("disruption_severity") > 0).cast("int"))

# ------------------------------------------------------------- derived columns
lines = lines.withColumn("order_year", F.year("order_ts")) \
    .withColumn("order_month", F.month("order_ts")) \
    .withColumn("order_quarter", F.quarter("order_ts")) \
    .withColumn("order_dow", F.dayofweek("order_ts")) \
    .withColumn("order_ym", F.date_format("order_ts", "yyyy-MM")) \
    .withColumn("is_canceled", F.col("delivery_status") == "Shipping canceled") \
    .withColumn("delay_days", F.col("days_real") - F.col("days_scheduled"))

with Timer("Writing curated Parquet (partitioned by order_year)"):
    lines.repartition("order_year").write.mode("overwrite").partitionBy("order_year").parquet(CURATED)

with Timer("Checking curated table"):
    cur = spark.read.parquet(CURATED)
    n_after = cur.count()
    stats = cur.agg(F.avg("late_delivery_risk").alias("late_rate"),
                    F.avg("disruption_flag").alias("disrupted_share"),
                    F.min("order_ts").alias("first_order"), F.max("order_ts").alias("last_order")).first().asDict()
    cur.printSchema()
    summary = {"raw_order_lines": n_before, "curated_order_lines": n_after, "dropped": n_before - n_after,
               "columns": len(cur.columns), **stats, "path": CURATED}
    print(summary)
    save_json(summary, "etl", "etl_summary.json")
spark.stop()
