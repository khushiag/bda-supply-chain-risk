"""
Objective 3b - Spark SQL: where do late deliveries pile up, and what drives them?

Every query runs distributed on YARN over the full curated table; only the small aggregated
answers are collected and saved to /workspace/output/analysis/*.csv (used for charts + report).
Cancelled orders are excluded from late-rate calculations (they never ship).

Run:  spark-submit 03_analysis.py
"""
from common import CURATED, Timer, get_spark, save_json

spark = get_spark("03-sparksql-risk-analysis")
spark.read.parquet(CURATED).createOrReplaceTempView("lines")
spark.sql("CREATE OR REPLACE TEMP VIEW shipped AS SELECT * FROM lines WHERE NOT is_canceled")
spark.catalog.cacheTable("shipped")

QUERIES = {
    "01_overall_kpis": """
        SELECT COUNT(*) AS order_lines, COUNT(DISTINCT order_id) AS orders,
               ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(AVG(delay_days), 3) AS avg_delay_days,
               ROUND(SUM(sales) / 1e6, 2) AS sales_musd,
               ROUND(SUM(order_profit_per_order) / 1e6, 2) AS profit_musd
        FROM shipped""",

    "02_by_shipping_mode": """
        SELECT shipping_mode, FIRST(days_scheduled) AS promised_days,
               COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(AVG(days_real), 2) AS avg_actual_days,
               ROUND(AVG(CASE WHEN late_delivery_risk = 1 THEN delay_days END), 2) AS avg_delay_when_late
        FROM shipped GROUP BY shipping_mode ORDER BY late_pct DESC""",

    "03_by_market": """
        SELECT market, COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(SUM(order_profit_per_order) / SUM(order_item_total) * 100, 2) AS margin_pct
        FROM shipped GROUP BY market ORDER BY late_pct DESC""",

    "04_by_region": """
        SELECT market, order_region, COUNT(*) AS lines,
               ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(AVG(port_congestion_index), 3) AS avg_congestion
        FROM shipped GROUP BY market, order_region ORDER BY late_pct DESC""",

    "05_heatmap_region_mode": """
        SELECT order_region, shipping_mode, COUNT(*) AS lines,
               ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct
        FROM shipped GROUP BY order_region, shipping_mode""",

    # Window function: rank carriers inside their market
    "06_by_carrier": """
        SELECT *, RANK() OVER (PARTITION BY market ORDER BY late_pct DESC) AS worst_in_market
        FROM (SELECT market, carrier_id, carrier_name, ROUND(FIRST(carrier_reliability), 3) AS reliability,
                     COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct
              FROM shipped GROUP BY market, carrier_id, carrier_name) t
        ORDER BY market, worst_in_market""",

    "07_by_supplier": """
        SELECT supplier_id, supplier_country, FIRST(base_lead_time_days) AS base_lead_days,
               ROUND(AVG(supplier_lead_time_days), 1) AS avg_lead_days, COUNT(*) AS lines,
               ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct
        FROM shipped GROUP BY supplier_id, supplier_country ORDER BY late_pct DESC""",

    "08_by_category_top20": """
        SELECT category_name, COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(SUM(sales) / 1e6, 2) AS sales_musd
        FROM shipped GROUP BY category_name ORDER BY lines DESC LIMIT 20""",

    "09_monthly_trend": """
        SELECT order_ym, COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct,
               ROUND(AVG(port_congestion_index), 3) AS avg_congestion,
               ROUND(AVG(weather_severity), 3) AS avg_weather
        FROM shipped GROUP BY order_ym ORDER BY order_ym""",

    "10_weather_x_congestion": """
        SELECT CASE WHEN weather_severity < 0.33 THEN '1 low' WHEN weather_severity < 0.66 THEN '2 medium' ELSE '3 high' END AS weather,
               CASE WHEN port_congestion_index < 0.33 THEN '1 low' WHEN port_congestion_index < 0.66 THEN '2 medium' ELSE '3 high' END AS congestion,
               COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct
        FROM shipped GROUP BY 1, 2 ORDER BY 1, 2""",

    "11_by_disruption": """
        SELECT disruption_type, disruption_severity, COUNT(*) AS lines,
               ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct, ROUND(AVG(delay_days), 2) AS avg_delay_days
        FROM shipped GROUP BY disruption_type, disruption_severity ORDER BY late_pct DESC""",

    "12_profit_impact": """
        SELECT delivery_status, COUNT(*) AS lines,
               ROUND(AVG(order_item_profit_ratio) * 100, 2) AS avg_profit_ratio_pct,
               ROUND(SUM(order_profit_per_order) / 1e6, 3) AS profit_musd
        FROM lines GROUP BY delivery_status ORDER BY lines DESC""",

    "13_yearly_mode_trend": """
        SELECT order_year, shipping_mode, COUNT(*) AS lines, ROUND(AVG(late_delivery_risk) * 100, 2) AS late_pct
        FROM shipped GROUP BY order_year, shipping_mode ORDER BY order_year, shipping_mode""",
}

timings = {}
for name, sql in QUERIES.items():
    with Timer(f"Query {name}") as t:
        df = spark.sql(sql)
        pdf = df.toPandas()
        pdf.to_csv(__import__("common").out_path("analysis", f"{name}.csv"), index=False)
    timings[name] = t.seconds
    print(pdf.head(12).to_string(index=False))

save_json(timings, "analysis", "query_timings_seconds.json")
spark.stop()
