"""
Charts for the report. Reads the small CSV/JSON results in /workspace/output and writes PNGs to
/workspace/output/figures (visible on Windows in hadoop-docker/workspace/output/figures).
Plain Python (no Spark needed):  python3 06_visualize.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.environ.get("BDA_OUT", "/workspace/output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
A = lambda n: os.path.join(OUT, "analysis", n)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})
BLUE, ORANGE, GREY = "#2a6f97", "#e07a2f", "#9aa5b1"
MODE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]
made = []


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close()
    made.append(name)


def exists(path):
    if not os.path.exists(path):
        print("   skip (missing):", path)
        return False
    return True


# 1. Seed vs generated validation
if exists(os.path.join(OUT, "validation", "seed_vs_generated.csv")):
    v = pd.read_csv(os.path.join(OUT, "validation", "seed_vs_generated.csv"))
    v = v[v.unit == "pp"]
    y = np.arange(len(v))
    plt.figure(figsize=(8, 5))
    plt.barh(y - 0.2, v.seed * 100, 0.4, label="Seed (DataCo)", color=GREY)
    plt.barh(y + 0.2, v.generated * 100, 0.4, label="Generated (DataCo-Scaled)", color=BLUE)
    plt.yticks(y, v.metric.str.replace("::", ": "))
    plt.xlabel("%"); plt.title("Seed vs generated distributions (all within 2 pp)" if v.within_2.all() else "Seed vs generated distributions")
    plt.legend(frameon=False); plt.gca().invert_yaxis()
    save("01_seed_vs_generated.png")

# 2. Late % by shipping mode
if exists(A("02_by_shipping_mode.csv")):
    m = pd.read_csv(A("02_by_shipping_mode.csv")).set_index("shipping_mode").reindex(MODE_ORDER).dropna(how="all").reset_index()
    plt.figure(figsize=(6, 3.6))
    bars = plt.bar(m.shipping_mode, m.late_pct, color=[ORANGE if x > 60 else BLUE for x in m.late_pct])
    plt.bar_label(bars, fmt="%.1f%%", padding=2)
    for i, d in enumerate(m.promised_days):
        plt.text(i, 3, f"promise {int(d)}d", ha="center", color="white", fontsize=8)
    plt.ylabel("Late deliveries (%)"); plt.title("Late-delivery rate by shipping mode")
    save("02_late_by_shipping_mode.png")

# 3. Late % by market
if exists(A("03_by_market.csv")):
    m = pd.read_csv(A("03_by_market.csv")).sort_values("late_pct")
    plt.figure(figsize=(6, 3.2))
    bars = plt.barh(m.market, m.late_pct, color=BLUE)
    plt.bar_label(bars, fmt="%.1f%%", padding=2)
    plt.xlim(m.late_pct.min() - 8, m.late_pct.max() + 4)
    plt.xlabel("Late deliveries (%)"); plt.title("Late-delivery rate by market")
    save("03_late_by_market.png")

# 4. Heatmap region x shipping mode
if exists(A("05_heatmap_region_mode.csv")):
    h = pd.read_csv(A("05_heatmap_region_mode.csv")).pivot(index="order_region", columns="shipping_mode", values="late_pct")
    h = h.reindex(columns=[c for c in MODE_ORDER if c in h.columns])
    h = h.loc[h.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(h) + 1.5))
    im = ax.imshow(h.values, cmap="YlOrRd", aspect="auto", vmin=20, vmax=100)
    ax.set_xticks(range(len(h.columns)), h.columns); ax.set_yticks(range(len(h)), h.index)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            val = h.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="white" if val > 75 else "black")
    fig.colorbar(im, ax=ax, label="Late %"); ax.set_title("Risk heatmap: late % by region and shipping mode")
    save("04_heatmap_region_mode.png")

# 5. Carriers
if exists(A("06_by_carrier.csv")):
    c = pd.read_csv(A("06_by_carrier.csv"))
    plt.figure(figsize=(6, 4))
    plt.scatter(c.reliability, c.late_pct, s=c.lines / c.lines.max() * 300, color=BLUE, alpha=.7)
    for _, r in c.iterrows():
        plt.annotate(r.carrier_id, (r.reliability, r.late_pct), fontsize=7, xytext=(4, 2), textcoords="offset points")
    plt.xlabel("Carrier on-time reliability"); plt.ylabel("Late deliveries (%)")
    plt.title("Carrier reliability vs late rate (bubble = volume)")
    save("05_carriers.png")

# 6. Monthly trend
if exists(A("09_monthly_trend.csv")):
    t = pd.read_csv(A("09_monthly_trend.csv"))
    t["date"] = pd.to_datetime(t.order_ym)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(t.date, t.late_pct, color=BLUE, lw=1.2, label="Late %")
    ax.set_ylabel("Late %"); ax2 = ax.twinx()
    ax2.plot(t.date, t.avg_congestion, color=ORANGE, lw=1, alpha=.8, label="Port congestion")
    ax2.set_ylabel("Avg port congestion"); ax2.spines["top"].set_visible(False)
    ax.set_title("Monthly late-delivery rate vs port congestion (2015-2025)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88), frameon=False)
    save("06_monthly_trend.png")

# 7. Weather x congestion
if exists(A("10_weather_x_congestion.csv")):
    w = pd.read_csv(A("10_weather_x_congestion.csv")).pivot(index="weather", columns="congestion", values="late_pct")
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(w.values, cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(w.shape[1]), [x[2:] for x in w.columns]); ax.set_yticks(range(w.shape[0]), [x[2:] for x in w.index])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            ax.text(j, i, f"{w.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Port congestion"); ax.set_ylabel("Weather severity")
    ax.set_title("Late % by weather x congestion"); fig.colorbar(im, ax=ax)
    save("07_weather_congestion.png")

# 8. Disruptions
if exists(A("11_by_disruption.csv")):
    d = pd.read_csv(A("11_by_disruption.csv"))
    d = d.groupby("disruption_type").apply(lambda g: pd.Series({"late_pct": np.average(g.late_pct, weights=g.lines), "lines": g.lines.sum()})).sort_values("late_pct")
    plt.figure(figsize=(6, 3.4))
    bars = plt.barh(d.index, d.late_pct, color=[GREY if i == "None" else ORANGE for i in d.index])
    plt.bar_label(bars, fmt="%.1f%%", padding=2)
    plt.xlabel("Late deliveries (%)"); plt.title("Late rate during disruption events")
    save("08_disruptions.png")

# 9-11. Models
ML = os.path.join(OUT, "ml")
if exists(os.path.join(ML, "model_comparison.csv")):
    mc = pd.read_csv(os.path.join(ML, "model_comparison.csv"))
    metrics = ["test_precision", "test_recall", "test_f1", "test_accuracy", "auc"]
    x = np.arange(len(metrics)); wdt = 0.8 / len(mc)
    plt.figure(figsize=(8, 3.8))
    for i, (_, r) in enumerate(mc.iterrows()):
        bars = plt.bar(x + i * wdt - 0.4 + wdt / 2, [r[m] for m in metrics], wdt, label=r.model,
                       color=[BLUE, ORANGE, GREY][i % 3])
        plt.bar_label(bars, fmt="%.3f", fontsize=6, padding=1)
    plt.axhline(0.80, ls="--", color="black", lw=.8); plt.text(len(metrics) - 0.5, 0.805, "F1 target 0.80", fontsize=7, ha="right")
    plt.xticks(x, ["Precision", "Recall", "F1", "Accuracy", "ROC-AUC"]); plt.ylim(0.5, 1.0)
    plt.legend(frameon=False, ncol=3, loc="upper left"); plt.title("Model comparison on unseen test data")
    save("09_model_comparison.png")

    plt.figure(figsize=(4.6, 4.4))
    for i, (_, r) in enumerate(mc.iterrows()):
        roc = pd.read_csv(os.path.join(ML, f"roc_{r.key}.csv"))
        plt.plot(roc.fpr, roc.tpr, color=[BLUE, ORANGE, GREY][i % 3], label=f"{r.model} (AUC {r.auc:.3f})")
    plt.plot([0, 1], [0, 1], ls=":", color="black", lw=.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
    plt.legend(frameon=False, fontsize=7, loc="lower right")
    save("10_roc_curves.png")

    fig, axes = plt.subplots(1, len(mc), figsize=(3.4 * len(mc), 3.2))
    for ax, (_, r) in zip(np.atleast_1d(axes), mc.iterrows()):
        cm = np.array([[r.test_tn, r.test_fp], [r.test_fn, r.test_tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm.sum() * 100:.1f}%)", ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["On time", "Late"]); ax.set_yticks([0, 1], ["On time", "Late"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{r.model}\nF1 {r.test_f1:.3f} @ t={r.threshold:.2f}", fontsize=8)
    save("11_confusion_matrices.png")

for key in ["gbt", "rf"]:
    f = os.path.join(ML, f"feature_importance_{key}.csv")
    if os.path.exists(f):
        fi = pd.read_csv(f).head(15).iloc[::-1]
        plt.figure(figsize=(6, 4.2))
        plt.barh(fi.feature, fi.importance, color=BLUE)
        plt.title(f"Top drivers of late delivery ({'GBT' if key == 'gbt' else 'Random Forest'} importance)")
        save(f"12_feature_importance_{key}.png")

# 13-14. Optimisation
OPT = os.path.join(OUT, "optimize")
if exists(os.path.join(OPT, "recommendations.csv")):
    r = pd.read_csv(os.path.join(OPT, "recommendations.csv"))
    if len(r):
        r["seg"] = r.order_region + "\n" + r.current_mode + " -> " + r.alt_mode
        r = r.sort_values("avoided_late_lines")
        plt.figure(figsize=(7.5, 0.45 * len(r) + 1.2))
        bars = plt.barh(r.seg, r.avoided_late_lines, color=BLUE)
        plt.bar_label(bars, labels=[f"{a:,.0f}  (margin {m:+.1f} pp)" for a, m in zip(r.avoided_late_lines, r.margin_change_pp)], fontsize=7, padding=2)
        plt.xlabel("Late deliveries avoided (model estimate)"); plt.yticks(fontsize=7)
        plt.title("What-if: switching shipping mode in the riskiest segments")
        save("13_mode_switch_recommendations.png")
if exists(os.path.join(OPT, "schedule_buffer.csv")):
    b = pd.read_csv(os.path.join(OPT, "schedule_buffer.csv")).set_index("shipping_mode").reindex(MODE_ORDER).dropna(how="all")
    x = np.arange(len(b))
    plt.figure(figsize=(6.5, 3.6))
    for i, (col, lab, colr) in enumerate([("late_pct_now", "Current promise", ORANGE),
                                          ("late_pct_with_1day_buffer", "+1 day buffer", BLUE),
                                          ("late_pct_with_2day_buffer", "+2 day buffer", GREY)]):
        bars = plt.bar(x + (i - 1) * 0.27, b[col], 0.27, label=lab, color=colr)
        plt.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    plt.xticks(x, b.index); plt.ylabel("Late deliveries (%)"); plt.legend(frameon=False)
    plt.title("Scheduling lever: effect of padding the promised date")
    save("14_schedule_buffer.png")

print(f"Saved {len(made)} charts to {FIG}:")
for m in made:
    print("  ", m)
