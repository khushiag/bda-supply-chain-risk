"""Smoke test: writes data to HDFS, reads it back with Spark SQL, trains a tiny MLlib model.
Run inside master:  spark-submit /workspace/test_cluster.py
"""
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("cluster-smoke-test").getOrCreate()

df = spark.range(0, 200_000).withColumn("x", F.rand(seed=1) * 100) \
          .withColumn("y", F.col("x") * 3 + F.randn(seed=2))
df.write.mode("overwrite").parquet("hdfs://master:9000/user/root/smoke")

back = spark.read.parquet("hdfs://master:9000/user/root/smoke")
back.createOrReplaceTempView("t")
spark.sql("SELECT COUNT(*) AS rows, ROUND(AVG(y), 2) AS avg_y FROM t").show()   # Spark SQL

data = VectorAssembler(inputCols=["x"], outputCol="features").transform(back)
model = LinearRegression(labelCol="y").fit(data)                                # MLlib
print(f">> slope learned: {model.coefficients[0]:.3f} (expected ~3.0)")
print(f">> executors used: {spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1}")

spark.stop()
