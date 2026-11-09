# Supply Chain Risk Detection and Delivery Optimization

Big Data Analytics (DSE 3125) mini project, Semester V, DSE-B.

## Overview

This project builds an end-to-end big data pipeline on a 3-node Hadoop cluster to find where late deliveries happen in a global supply chain, predict them at the time an order is placed, and suggest changes that reduce them.

- Storage: HDFS
- Compute: Apache Spark on YARN (PySpark, client mode)
- Spark components used: Spark SQL / DataFrames and Spark MLlib

## Cluster

Three Docker containers act as three nodes:

- master: NameNode, SecondaryNameNode, ResourceManager, Spark History Server
- worker1: DataNode, NodeManager
- worker2: DataNode, NodeManager

Versions: Hadoop 3.4.1, Spark 3.5.3, Java 11, Python 3.10.
HDFS replication factor is 2. Each NodeManager offers 2.5 GB memory and 2 vcores.

## Dataset

Seed: DataCo Smart Supply Chain for Big Data Analysis (Mendeley Data, CC BY 4.0), 180,519 order lines.
https://data.mendeley.com/datasets/8gx2fvg2k6/5

## Repository structure

    docker-compose.yml        3-node cluster definition
    Dockerfile                image with Hadoop, Spark, Java, Python
    entrypoint.sh             starts the correct daemons for master / worker
    config/                   Hadoop and Spark configuration files
    workspace/
      test_cluster.py         small job to check the cluster works
      data/                   place the seed CSV here
      supply_chain/           pipeline code (stages 01 to 06)
      output/                 results from our run (CSV, JSON, charts, logs)

## Pipeline

    01_generate.py   scale the seed to ~3M lines, write 5 CSV tables to HDFS, validate against seed
    02_etl.py        clean, join tables (broadcast and range joins), write Parquet partitioned by year
    03_analysis.py   13 Spark SQL queries on late-delivery risk
    04_train.py      train Logistic Regression, Random Forest, GBT with MLlib; save best model to HDFS
    05_optimize.py   riskiest segments, shipping-mode what-if analysis, scheduling buffer analysis
    06_visualize.py  charts
    run_all.sh       runs all stages on YARN
    common.py        shared paths and helpers

HDFS layout:

    /dataco/seed                  original CSV
    /dataco/raw                   generated tables (orders, shipments, carriers, suppliers, disruptions)
    /dataco/curated/order_lines   cleaned and joined Parquet
    /dataco/models/best_model     saved MLlib pipeline model

## How to run

Requirements: Docker Desktop with the WSL 2 engine, about 8 GB RAM given to WSL.

1. Put DataCoSupplyChainDataset.csv in workspace/data/.

2. Build and start the cluster (first build takes 5 to 15 minutes):

       docker compose up -d --build
       docker compose ps

3. Open a shell in the master container:

       docker exec -it master bash

4. Check the cluster:

       hdfs dfsadmin -report | grep "Live datanodes"
       yarn node -list

5. Run the full pipeline (about 10 to 15 minutes):

       bash /workspace/supply_chain/run_all.sh

   To run only some stages:

       bash /workspace/supply_chain/run_all.sh 03 05 06

6. Results are written to workspace/output/.

Web UIs:

    HDFS NameNode         http://localhost:9870
    YARN ResourceManager  http://localhost:8088
    Spark History Server  http://localhost:18080

To stop the cluster: docker compose stop
To start it again:   docker compose start

## Results

Data validation:
- Late rate, shipping mode mix and market mix all within 2% of the seed

Analysis (shipped orders only, 2.87M lines):
- Overall late-delivery rate: 57.3%
- By shipping mode: First Class 100% late (promises 1 day, takes about 2.5), Second Class 79.8%, Same Day 47.9%, Standard Class 39.8%
- Worst carrier in each market is about 70% late, best is about 40%
- Low weather and low congestion: 40% late. High weather and high congestion: 99.9% late
- Severe disruptions (storms, port closures, lockdowns): about 99% late

Machine learning (test set, 578,902 rows):

    Model                    AUC     Precision  Recall   F1
    Logistic Regression      0.892   0.813      0.850    0.831
    Gradient-Boosted Trees   0.891   0.803      0.859    0.830
    Random Forest            0.882   0.785      0.869    0.825

Best model: Logistic Regression, F1 0.831 (target was 0.80).

Optimisation:
- Switching the 12 riskiest segments to Standard Class avoids about 274,000 late lines (16.7% of all late lines). This is mainly because the promised delivery time becomes longer.
- Adding one day of buffer to promised delivery dates would make about 878,000 late lines on time.

## Output files

    workspace/output/validation/   seed vs generated comparison
    workspace/output/etl/          ETL summary
    workspace/output/analysis/     results of the 13 Spark SQL queries
    workspace/output/ml/           model comparison, ROC data, feature importance, best model info
    workspace/output/optimize/     risky segments, what-if results, recommendations, buffer analysis
    workspace/output/figures/      charts
    workspace/output/logs/         logs from each stage

## Reference

F. Constante, F. Silva and A. Pereira, "DataCo Smart Supply Chain for Big Data Analysis",
Mendeley Data, V5, 2019. doi: 10.17632/8gx2fvg2k6.5