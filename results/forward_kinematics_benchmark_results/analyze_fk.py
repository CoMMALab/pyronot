"""Analyze FK benchmark results: produce a LaTeX-ready table and scaling plot."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
df = pd.read_csv(OUT_DIR / "fk_benchmark_aggregated.csv")

# ── 1. LaTeX table: time (ms) per method × batch size, one sub-table per robot ──

METHOD_ORDER = ["pyroki", "pyronot", "curobo"]
ROBOT_ORDER = ["panda", "fetch", "baxter", "g1"]
ROBOT_LABELS = {"panda": "Panda (7-DOF)", "fetch": "Fetch (8-DOF)",
                "baxter": "Baxter (15-DOF)", "g1": "G1 (37-DOF)"}

pivot = (
    df.pivot_table(index=["robot", "batch_size"], columns="method", values="time_ms")
    .reindex(columns=METHOD_ORDER)
)

def fmt(v):
    if pd.isna(v):
        return "---"
    if v < 0.01:
        return f"{v*1000:.1f}\\,\\textmu s"
    if v < 1:
        return f"{v:.3f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Forward kinematics compute time (ms) vs.\ batch size.}")
lines.append(r"\label{tab:fk_benchmark}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{r" + "c" * len(METHOD_ORDER) + "}")
lines.append(r"\toprule")
lines.append(r"Batch & " + " & ".join(m.replace("_", r"\_") for m in METHOD_ORDER) + r" \\")

for robot in ROBOT_ORDER:
    if robot not in pivot.index.get_level_values(0):
        continue
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(METHOD_ORDER) + 1) + "}{l}{\\textbf{" + ROBOT_LABELS[robot] + r"}} \\")
    sub = pivot.loc[robot]
    for batch, row in sub.iterrows():
        cells = [f"{batch:,}"] + [fmt(row.get(m, np.nan)) for m in METHOD_ORDER]
        lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
(OUT_DIR / "fk_table.tex").write_text(latex_str)
print("Wrote fk_table.tex")
print(latex_str)

# ── 2. Scaling plot ──

fig, axes = plt.subplots(1, len(ROBOT_ORDER), figsize=(14, 3.5), sharey=True)

COLORS = {"pyroki": "#4C72B0", "pyronot": "#DD5544", "curobo": "#55A868"}
MARKERS = {"pyroki": "o", "pyronot": "s", "curobo": "^"}

for ax, robot in zip(axes, ROBOT_ORDER):
    rdf = df[df["robot"] == robot]
    for method in METHOD_ORDER:
        mdf = rdf[rdf["method"] == method].sort_values("batch_size")
        if mdf.empty:
            continue
        ax.plot(mdf["batch_size"], mdf["time_ms"],
                marker=MARKERS[method], markersize=5, linewidth=1.5,
                color=COLORS[method], label=method)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(ROBOT_LABELS[robot], fontsize=10)
    ax.set_xlabel("Batch size")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"$2^{{{int(np.log2(x))}}}$" if x >= 1 and np.log2(x) == int(np.log2(x)) else ""))
    all_batches = sorted(rdf["batch_size"].unique())
    ax.set_xticks(all_batches[::2])
    ax.tick_params(axis="x", labelsize=8, rotation=0)

axes[0].set_ylabel("Time (ms)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(METHOD_ORDER),
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_DIR / "fk_scaling.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fk_scaling.png", dpi=200, bbox_inches="tight")
print("Wrote fk_scaling.pdf and fk_scaling.png")
