"""Analyze IK benchmark results: aggregate, produce LaTeX tables and plots."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ── 0. Load and aggregate ──────────────────────────────────────────────────────

def strip_batch_suffix(solver: str) -> str:
    return solver.removesuffix("-BATCH")


df_pyronot = pd.read_csv(OUT_DIR / "bench_ik_results.csv")
df_curobo  = pd.read_csv(OUT_DIR / "bench_ik_results_curobo.csv")

df_pyronot["source"] = "pyronot"
df_curobo["source"]  = "curobo"

df = pd.concat([df_pyronot, df_curobo], ignore_index=True)

df["solver_key"] = df["solver"].apply(strip_batch_suffix)
df["success_rate"] = df["success_n"] / df["success_total"] * 100
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Deduplicate: keep most recent run per (robot, mode, solver_key, collision_free)
df = (df.sort_values("timestamp")
        .drop_duplicates(subset=["robot", "mode", "solver_key", "collision_free"], keep="last")
        .reset_index(drop=True))

df.to_csv(OUT_DIR / "ik_benchmark_aggregated.csv", index=False)
print(f"Wrote ik_benchmark_aggregated.csv  ({len(df)} rows)")

# ── Shared config ──────────────────────────────────────────────────────────────

ROBOTS        = ["panda", "fetch", "baxter"]
ROBOT_LABELS  = {"panda": "Panda (7-DOF)", "fetch": "Fetch (8-DOF)", "baxter": "Baxter (15-DOF)"}
ROBOT_SHORT   = {"panda": "Panda",         "fetch": "Fetch",          "baxter": "Baxter"}

# Canonical display order for tables
SEQ_SOLVER_ORDER = [
    "HJCD-JAX", "HJCD-CUDA",
    "LS-JAX",   "LS-CUDA",
    "SQP-JAX",  "SQP-CUDA",
    "MPPI-JAX", "MPPI-CUDA",
    "PyRoki",
    "cuRobo",
]

BATCH_SOLVER_ORDER = [
    "HJCD-JAX", "HJCD-CUDA",
    "LS-JAX",   "LS-CUDA",
    "SQP-JAX",  "SQP-CUDA",
    "MPPI-JAX", "MPPI-CUDA",
    "PyRoki",
    "cuRobo",
]

SOLVER_DISPLAY = {
    "HJCD-JAX":  r"PyRoNot-HJCD (JAX)",
    "HJCD-CUDA": r"PyRoNot-HJCD (CUDA)",
    "LS-JAX":    r"PyRoNot-LS (JAX)",
    "LS-CUDA":   r"PyRoNot-LS (CUDA)",
    "SQP-JAX":   r"PyRoNot-SQP (JAX)",
    "SQP-CUDA":  r"PyRoNot-SQP (CUDA)",
    "MPPI-JAX":  r"PyRoNot-MPPI (JAX)",
    "MPPI-CUDA": r"PyRoNot-MPPI (CUDA)",
    "Learned-JAX": r"Learned (JAX)",
    "PyRoki":    r"PyRoki",
    "cuRobo":    r"cuRobo",
}

# Algorithm family groups for visual midrules
ALGO_FAMILIES = [
    ["HJCD-JAX", "HJCD-CUDA"],
    ["LS-JAX",   "LS-CUDA"],
    ["SQP-JAX",  "SQP-CUDA"],
    ["MPPI-JAX", "MPPI-CUDA"],
    ["PyRoki"],
    ["cuRobo"],
]


def fmt_time(v, *, bold=False):
    if pd.isna(v):
        return "---"
    if v < 0.01:
        s = f"{v*1000:.1f}\\,\\textmu s"
    elif v < 1:
        s = f"{v:.3f}"
    elif v < 100:
        s = f"{v:.2f}"
    else:
        s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_pct(v):
    if pd.isna(v):
        return "---"
    return f"{v:.0f}\\%"


def fmt_pos(v):
    if pd.isna(v):
        return "---"
    if v < 0.001:
        return f"{v*1000:.2f}\\,\\textmu m"
    if v < 1.0:
        return f"{v:.3f}"
    return f"{v:.1f}"


def fmt_rot(v):
    if pd.isna(v):
        return "---"
    if v < 1e-4:
        return f"<0.0001"
    return f"{v:.4f}"


def best_in_col(series):
    """Return the index of the minimum finite value."""
    valid = series.dropna()
    return valid.idxmin() if not valid.empty else None


# ── Table 1: Sequential IK latency — no collision ─────────────────────────────
#
# Rows: solver | Panda t_med / t_p95 | Fetch ... | Baxter ...
# Only sequential, collision_free=False

print("\n── Table 1: Sequential IK latency (no collision) ──")

sub = df[(df["mode"] == "sequential") & (df["collision_free"] == False)]

# Build pivot: index=solver_key, columns=robot, values=(t_med_ms, t_p95_ms)
piv = sub.pivot_table(index="solver_key", columns="robot",
                      values=["t_med_ms", "t_p95_ms"], aggfunc="first")

lines = []
n_robots = len(ROBOTS)
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Sequential IK latency with no collision avoidance "
             r"(10 problems; 5 timed runs). "
             r"Reported as median\,/\,p95 (ms). "
             r"Bold: fastest median per robot.}")
lines.append(r"\label{tab:ik_seq_nocoll}")
lines.append(r"\small")
# cols: Solver + 2 cols per robot (med / p95)
col_spec = "l" + "".join(["cc"] * n_robots)
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")
# Header row 1: robot spans
header1 = r"\multicolumn{1}{c}{} & "
header1 += " & ".join(
    r"\multicolumn{2}{c}{\textbf{" + ROBOT_SHORT[r] + r"}}" for r in ROBOTS
)
header1 += r" \\"
lines.append(header1)
lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
# Header row 2: Solver | med | p95 repeated
header2 = r"\textbf{Solver} & "
header2 += " & ".join([r"med & p95"] * n_robots)
header2 += r" \\"
lines.append(header2)
lines.append(r"\midrule")

# Find best (lowest) median per robot
best_med = {}
for robot in ROBOTS:
    col = ("t_med_ms", robot)
    if col in piv.columns:
        vals = {sk: piv.loc[sk, col] for sk in SEQ_SOLVER_ORDER if sk in piv.index and not pd.isna(piv.loc[sk, col])}
        if vals:
            best_med[robot] = min(vals, key=vals.get)

first_family = True
for family in ALGO_FAMILIES:
    if not first_family:
        lines.append(r"\midrule")
    first_family = False
    for sk in family:
        if sk not in piv.index:
            continue
        cells = [SOLVER_DISPLAY.get(sk, sk)]
        for robot in ROBOTS:
            med_col = ("t_med_ms", robot)
            p95_col = ("t_p95_ms", robot)
            med = piv.loc[sk, med_col] if med_col in piv.columns else np.nan
            p95 = piv.loc[sk, p95_col] if p95_col in piv.columns else np.nan
            is_best = (best_med.get(robot) == sk)
            cells.append(fmt_time(med, bold=is_best))
            cells.append(fmt_time(p95))
        lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table1 = "\n".join(lines)
(OUT_DIR / "ik_table1_seq_nocoll.tex").write_text(table1)
print(table1)


# ── Table 2: Batch IK throughput — no collision ────────────────────────────────
#
# Rows: solver | t_med_ms per robot | VRAM (MB) per robot
# Only batch, collision_free=False, 256 problems

print("\n── Table 2: Batch IK throughput (no collision, 256 problems) ──")

sub2 = df[(df["mode"] == "batch") & (df["collision_free"] == False)]

piv2 = sub2.pivot_table(index="solver_key", columns="robot",
                        values=["t_med_ms", "peak_vram_mb"], aggfunc="first")

best_batch_med = {}
for robot in ROBOTS:
    col = ("t_med_ms", robot)
    if col in piv2.columns:
        vals = {sk: piv2.loc[sk, col]
                for sk in BATCH_SOLVER_ORDER
                if sk in piv2.index and not pd.isna(piv2.loc[sk, col])}
        if vals:
            best_batch_med[robot] = min(vals, key=vals.get)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Batch IK throughput with no collision avoidance "
             r"(256 simultaneous problems; 5 timed runs). "
             r"Time is the median wall-clock duration for the full batch (ms). "
             r"VRAM is the peak GPU memory during inference (MB). "
             r"Bold: fastest per robot.}")
lines.append(r"\label{tab:ik_batch_nocoll}")
lines.append(r"\small")
col_spec = "l" + "".join(["cc"] * n_robots)
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")
header1 = r"\multicolumn{1}{c}{} & "
header1 += " & ".join(
    r"\multicolumn{2}{c}{\textbf{" + ROBOT_SHORT[r] + r"}}" for r in ROBOTS
)
header1 += r" \\"
lines.append(header1)
cmidrule = r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
lines.append(cmidrule)
header2 = r"\textbf{Solver} & "
header2 += " & ".join([r"time (ms) & VRAM (MB)"] * n_robots)
header2 += r" \\"
lines.append(header2)
lines.append(r"\midrule")

first_family = True
for family in ALGO_FAMILIES:
    if not first_family:
        lines.append(r"\midrule")
    first_family = False
    for sk in family:
        if sk not in piv2.index:
            continue
        cells = [SOLVER_DISPLAY.get(sk, sk)]
        for robot in ROBOTS:
            t_col   = ("t_med_ms",    robot)
            vram_col = ("peak_vram_mb", robot)
            t    = piv2.loc[sk, t_col]    if t_col    in piv2.columns else np.nan
            vram = piv2.loc[sk, vram_col] if vram_col in piv2.columns else np.nan
            is_best = (best_batch_med.get(robot) == sk)
            cells.append(fmt_time(t, bold=is_best))
            if pd.isna(vram):
                cells.append("---")
            else:
                cells.append(f"{int(vram):,}")
        lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table2 = "\n".join(lines)
(OUT_DIR / "ik_table2_batch_nocoll.tex").write_text(table2)
print(table2)


# ── Table 3: Collision-free IK — sequential ────────────────────────────────────
#
# For each (robot, solver): success%, t_med_ms, pos_med_mm, rot_med_rad
# Only sequential, collision_free=True

print("\n── Table 3: Collision-free IK (sequential) ──")

sub3 = df[(df["mode"] == "sequential") & (df["collision_free"] == True)]

CF_SOLVER_ORDER = [
    "HJCD-JAX", "HJCD-CUDA",
    "LS-JAX",   "LS-CUDA",
    "SQP-JAX",  "SQP-CUDA",
    "MPPI-JAX", "MPPI-CUDA",
    "PyRoki",
    "cuRobo",
]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Collision-free IK performance in sequential mode "
             r"(10 problems each). "
             r"Success: fraction of problems for which a collision-free solution was found. "
             r"Pos.\ error and rot.\ error are medians over \emph{all} attempted problems "
             r"(including failures, where the solver returns its best infeasible solution). "
             r"Timing is the median per-problem wall-clock time (ms).}")
lines.append(r"\label{tab:ik_cf_seq}")
lines.append(r"\small")
# cols: Solver | robot × (succ%, time, pos, rot)
n_cols_per_robot = 4
col_spec = "l" + "".join(["cccc"] * n_robots)
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")

header1 = r"\multicolumn{1}{c}{} & "
spans = []
start_col = 2
for r in ROBOTS:
    end_col = start_col + n_cols_per_robot - 1
    spans.append(
        r"\multicolumn{" + str(n_cols_per_robot) + r"}{c}{\textbf{" + ROBOT_SHORT[r] + r"}}"
    )
    start_col = end_col + 1
header1 += " & ".join(spans) + r" \\"
lines.append(header1)

cmidrule_parts = []
col = 2
for _ in ROBOTS:
    cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col + n_cols_per_robot - 1}}}")
    col += n_cols_per_robot
lines.append("".join(cmidrule_parts))

header2 = r"\textbf{Solver}"
for _ in ROBOTS:
    header2 += r" & Succ.\% & Time & Pos (mm) & Rot (rad)"
header2 += r" \\"
lines.append(header2)
lines.append(r"\midrule")

piv3 = sub3.pivot_table(
    index="solver_key", columns="robot",
    values=["success_rate", "t_med_ms", "pos_med_mm", "rot_med_rad"],
    aggfunc="first"
)

first_family = True
for family in ALGO_FAMILIES:
    if not first_family:
        lines.append(r"\midrule")
    first_family = False
    for sk in CF_SOLVER_ORDER:
        if sk not in [s for f in ALGO_FAMILIES for s in f]:
            continue
        if sk not in family:
            continue
        if sk not in piv3.index:
            continue
        cells = [SOLVER_DISPLAY.get(sk, sk)]
        for robot in ROBOTS:
            sr  = piv3.loc[sk, ("success_rate", robot)] if ("success_rate", robot) in piv3.columns else np.nan
            t   = piv3.loc[sk, ("t_med_ms",     robot)] if ("t_med_ms",     robot) in piv3.columns else np.nan
            pos = piv3.loc[sk, ("pos_med_mm",    robot)] if ("pos_med_mm",    robot) in piv3.columns else np.nan
            rot = piv3.loc[sk, ("rot_med_rad",   robot)] if ("rot_med_rad",   robot) in piv3.columns else np.nan
            cells.append(fmt_pct(sr))
            cells.append(fmt_time(t))
            cells.append(fmt_pos(pos))
            cells.append(fmt_rot(rot))
        lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table3 = "\n".join(lines)
(OUT_DIR / "ik_table3_cf_seq.tex").write_text(table3)
print(table3)


# ── Table 4: Collision-free IK — batch (256 problems) ─────────────────────────

print("\n── Table 4: Collision-free IK — batch (256 problems) ──")

sub4 = df[(df["mode"] == "batch") & (df["collision_free"] == True)]

piv4 = sub4.pivot_table(
    index="solver_key", columns="robot",
    values=["success_rate", "t_med_ms", "pos_med_mm"],
    aggfunc="first"
)

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Collision-free IK in batch mode (256 problems). "
             r"Success: fraction of the batch for which a collision-free solution was found. "
             r"Time is the median wall-clock duration for the \emph{full batch} (ms). "
             r"Pos.\ error is the median over all attempted problems (mm).}")
lines.append(r"\label{tab:ik_cf_batch}")
lines.append(r"\small")
n_cols_per_robot_b = 3
col_spec = "l" + "".join(["ccc"] * n_robots)
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")

header1 = r"\multicolumn{1}{c}{} & "
spans = [
    r"\multicolumn{3}{c}{\textbf{" + ROBOT_SHORT[r] + r"}}" for r in ROBOTS
]
header1 += " & ".join(spans) + r" \\"
lines.append(header1)

col = 2
parts = []
for _ in ROBOTS:
    parts.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
    col += 3
lines.append("".join(parts))

header2 = r"\textbf{Solver}"
for _ in ROBOTS:
    header2 += r" & Succ.\% & Time (ms) & Pos (mm)"
header2 += r" \\"
lines.append(header2)
lines.append(r"\midrule")

BATCH_CF_SOLVER_ORDER = [
    "HJCD-JAX", "HJCD-CUDA",
    "LS-JAX",   "LS-CUDA",
    "SQP-JAX",  "SQP-CUDA",
    "MPPI-JAX", "MPPI-CUDA",
    "PyRoki",
    "cuRobo",
]

first_family = True
for family in ALGO_FAMILIES:
    if not first_family:
        lines.append(r"\midrule")
    first_family = False
    for sk in family:
        if sk not in piv4.index:
            continue
        cells = [SOLVER_DISPLAY.get(sk, sk)]
        for robot in ROBOTS:
            sr  = piv4.loc[sk, ("success_rate", robot)] if ("success_rate", robot) in piv4.columns else np.nan
            t   = piv4.loc[sk, ("t_med_ms",     robot)] if ("t_med_ms",     robot) in piv4.columns else np.nan
            pos = piv4.loc[sk, ("pos_med_mm",    robot)] if ("pos_med_mm",    robot) in piv4.columns else np.nan
            cells.append(fmt_pct(sr))
            cells.append(fmt_time(t))
            cells.append(fmt_pos(pos))
        lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table4 = "\n".join(lines)
(OUT_DIR / "ik_table4_cf_batch.tex").write_text(table4)
print(table4)


# ── Table 5: CUDA vs JAX speedup (sequential, no collision) ───────────────────

print("\n── Table 5: CUDA speedup over JAX ──")

sub5 = df[(df["mode"] == "sequential") & (df["collision_free"] == False)]
piv5 = sub5.pivot_table(index="solver_key", columns="robot",
                        values="t_med_ms", aggfunc="first")

ALGO_PAIRS = [("HJCD-JAX", "HJCD-CUDA"), ("LS-JAX", "LS-CUDA"),
              ("SQP-JAX", "SQP-CUDA"),   ("MPPI-JAX", "MPPI-CUDA")]
ALGO_NAMES = {"HJCD": "PyRoNot-HJCD", "LS": "PyRoNot-LS", "SQP": "PyRoNot-SQP", "MPPI": "PyRoNot-MPPI"}

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{CUDA speedup factor over the equivalent JAX implementation "
             r"in sequential IK (no collision avoidance). "
             r"Speedup $= t_{\text{JAX}} / t_{\text{CUDA}}$.}")
lines.append(r"\label{tab:ik_cuda_speedup}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{l" + "c" * n_robots + "}")
lines.append(r"\toprule")
lines.append(r"\textbf{Algorithm} & " +
             " & ".join(r"\textbf{" + ROBOT_SHORT[r] + r"}" for r in ROBOTS) + r" \\")
lines.append(r"\midrule")

for jax_sk, cuda_sk in ALGO_PAIRS:
    algo_name = "PyRoNot " + jax_sk.replace("-JAX", "")
    cells = [algo_name]
    for robot in ROBOTS:
        t_jax  = piv5.loc[jax_sk,  robot] if (jax_sk  in piv5.index and robot in piv5.columns) else np.nan
        t_cuda = piv5.loc[cuda_sk, robot] if (cuda_sk in piv5.index and robot in piv5.columns) else np.nan
        if pd.isna(t_jax) or pd.isna(t_cuda) or t_cuda == 0:
            cells.append("---")
        else:
            speedup = t_jax / t_cuda
            cells.append(f"{speedup:.1f}$\\times$")
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table5 = "\n".join(lines)
(OUT_DIR / "ik_table5_cuda_speedup.tex").write_text(table5)
print(table5)


# ── Figure: batch time vs problem count — bar chart ───────────────────────────

print("\n── Figure: batch IK latency comparison (no collision, 256 problems) ──")

sub_fig = df[(df["mode"] == "batch") & (df["collision_free"] == False)]

PLOT_SOLVERS = ["HJCD-JAX", "HJCD-CUDA", "LS-JAX", "LS-CUDA",
                "SQP-JAX",  "SQP-CUDA",  "MPPI-JAX", "MPPI-CUDA",
                "PyRoki", "cuRobo"]
COLORS_JAX  = "#CC2288"
COLORS_CUDA = "#DD5544"
COLORS_OTHER = {"PyRoki": "#888888", "cuRobo": "#55A868"}

def solver_color(sk):
    if sk.endswith("-JAX"):
        return COLORS_JAX
    if sk.endswith("-CUDA"):
        return COLORS_CUDA
    return COLORS_OTHER.get(sk, "#AAAAAA")

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, robot in zip(axes, ROBOTS):
    rdf = sub_fig[sub_fig["robot"] == robot].set_index("solver_key")
    solvers_present = [s for s in PLOT_SOLVERS if s in rdf.index]
    times   = [rdf.loc[s, "t_med_ms"] for s in solvers_present]
    p95s    = [rdf.loc[s, "t_p95_ms"] - rdf.loc[s, "t_med_ms"]
               if not pd.isna(rdf.loc[s, "t_p95_ms"]) else 0
               for s in solvers_present]
    colors  = [solver_color(s) for s in solvers_present]
    labels  = [SOLVER_DISPLAY.get(s, s) for s in solvers_present]

    x = np.arange(len(solvers_present))
    bars = ax.bar(x, times, color=colors, width=0.7, zorder=2)
    ax.errorbar(x, times, yerr=[np.zeros(len(p95s)), p95s],
                fmt="none", color="black", capsize=3, linewidth=1, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_title(ROBOT_LABELS[robot], fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.4, zorder=0)
    ax.set_xlabel("")

axes[0].set_ylabel("Median time (ms)")

legend_elements = [
    Patch(facecolor=COLORS_JAX,  label="JAX backend"),
    Patch(facecolor=COLORS_CUDA, label="CUDA backend"),
    Patch(facecolor="#55A868",   label="cuRobo"),
    Patch(facecolor="#888888",   label="PyRoki"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03))

fig.suptitle("Batch IK Latency (256 Problems) — No Collision Avoidance", fontsize=11, y=1.05)
fig.tight_layout()
fig.savefig(OUT_DIR / "ik_seq_latency.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "ik_seq_latency.png", dpi=200, bbox_inches="tight")
print("Wrote ik_seq_latency.pdf/.png")
plt.close(fig)


# ── Figures: IK latency bar charts ────────────────────────────────────────────

def _latency_bar_fig(sub_df, title, out_stem, solvers=None):
    """Bar chart of median IK latency (log scale) across robots."""
    if solvers is None:
        solvers = PLOT_SOLVERS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, robot in zip(axes, ROBOTS):
        rdf = sub_df[sub_df["robot"] == robot].set_index("solver_key")
        present = [s for s in solvers if s in rdf.index]
        times  = [rdf.loc[s, "t_med_ms"] for s in present]
        p95s   = [rdf.loc[s, "t_p95_ms"] - rdf.loc[s, "t_med_ms"]
                  if not pd.isna(rdf.loc[s, "t_p95_ms"]) else 0
                  for s in present]
        colors = [solver_color(s) for s in present]
        labels = [SOLVER_DISPLAY.get(s, s) for s in present]
        x = np.arange(len(present))
        ax.bar(x, times, color=colors, width=0.7, zorder=2)
        ax.errorbar(x, times, yerr=[np.zeros(len(p95s)), p95s],
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(ROBOT_LABELS[robot], fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.4, zorder=0)
    axes[0].set_ylabel("Median time (ms)")
    legend_elements = [
        Patch(facecolor=COLORS_JAX,  label="JAX backend"),
        Patch(facecolor=COLORS_CUDA, label="CUDA backend"),
        Patch(facecolor="#55A868",   label="cuRobo"),
        Patch(facecolor="#4C72B0",   label="PyRoki"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out_stem}.pdf/.png")
    plt.close(fig)


print("\n── Figure: Sequential IK latency — no collision avoidance ──")
_latency_bar_fig(
    df[(df["mode"] == "sequential") & (df["collision_free"] == False)],
    title="Sequential IK Latency — No Collision Avoidance",
    out_stem="ik_latency_seq_nocoll",
)

print("\n── Figure: Sequential IK latency — with collision avoidance ──")
_latency_bar_fig(
    df[(df["mode"] == "sequential") & (df["collision_free"] == True)],
    title="Sequential IK Latency — Collision Avoidance",
    out_stem="ik_latency_seq_cf",
)

print("\n── Figure: Batch IK latency — no collision avoidance (256 problems) ──")
_latency_bar_fig(
    df[(df["mode"] == "batch") & (df["collision_free"] == False)],
    title="Batch IK Latency (256 Problems) — No Collision Avoidance",
    out_stem="ik_latency_batch_nocoll",
)

print("\n── Figure: Batch IK latency — collision avoidance (256 problems) ──")
_latency_bar_fig(
    df[(df["mode"] == "batch") & (df["collision_free"] == True)],
    title="Batch IK Latency (256 Problems) — Collision Avoidance",
    out_stem="ik_latency_batch_cf",
)


# ── Figure: Batch pos/rot error — CF vs no-CF ────────────────────────────────

print("\n── Figure: Batch pos/rot error comparison (CF vs no-CF) ──")

ERROR_SOLVERS = ["HJCD-JAX", "HJCD-CUDA", "LS-JAX", "LS-CUDA",
                 "SQP-JAX",  "SQP-CUDA",  "MPPI-JAX", "MPPI-CUDA",
                 "PyRoki", "cuRobo"]

sub_nocf = df[(df["mode"] == "batch") & (df["collision_free"] == False)]
sub_cf   = df[(df["mode"] == "batch") & (df["collision_free"] == True)]

ROBOT_COLORS  = {"panda": "#4CC9F0", "fetch": "#06D6A0", "baxter": "#F72585"}
ROBOT_OFFSETS = {"panda": -0.22, "fetch": 0.0, "baxter": 0.22}
EPSILON       = 1e-6   # floor for log scale

def _err_val(rdf, sk, col):
    v = rdf.loc[sk, col] if sk in rdf.index else np.nan
    return float(v) if not pd.isna(v) else np.nan

# y positions with extra gap between algorithm families
ordered_solvers = [sk for fam in ALGO_FAMILIES for sk in fam if sk in ERROR_SOLVERS]
y_pos = {}
y = 0.0
for i, fam in enumerate(ALGO_FAMILIES):
    if i > 0:
        y += 0.3
    for sk in fam:
        if sk in ERROR_SOLVERS:
            y_pos[sk] = y
            y += 0.7

ytick_vals   = [y_pos[sk] for sk in ordered_solvers]
ytick_labels = [SOLVER_DISPLAY.get(sk, sk) for sk in ordered_solvers]

metrics = [
    ("pos_med_mm",  "Median position error (mm)"),
    ("rot_med_rad", "Median rotation error (rad)"),
]

from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

for ax, (med_col, xlabel) in zip(axes, metrics):
    for sk in ordered_solvers:
        y_base = y_pos[sk]
        for robot in ROBOTS:
            color    = ROBOT_COLORS[robot]
            robot_y  = y_base + ROBOT_OFFSETS[robot]
            nocf_rdf = sub_nocf[sub_nocf["robot"] == robot].set_index("solver_key")
            cf_rdf   = sub_cf  [sub_cf  ["robot"] == robot].set_index("solver_key")

            v_nocf  = _err_val(nocf_rdf, sk, med_col)
            v_cf    = _err_val(cf_rdf,   sk, med_col)
            vp_nocf = max(v_nocf, EPSILON) if not np.isnan(v_nocf) else np.nan
            vp_cf   = max(v_cf,   EPSILON) if not np.isnan(v_cf)   else np.nan

            if not (np.isnan(vp_nocf) or np.isnan(vp_cf)):
                ax.plot([vp_nocf, vp_cf], [robot_y, robot_y],
                        color=color, alpha=0.35, linewidth=1.2, zorder=1)
            if not np.isnan(vp_nocf):
                ax.scatter([vp_nocf], [robot_y], color=color, marker="s",
                           s=55, zorder=3, linewidths=0)
            if not np.isnan(vp_cf):
                ax.scatter([vp_cf], [robot_y], color=color, marker="D",
                           s=45, zorder=4, linewidths=0)

    # Faint separator lines between algorithm families
    sep_y = 0.0
    for i, fam in enumerate(ALGO_FAMILIES):
        if i > 0:
            ax.axhline(sep_y - 0.15, color="#dddddd", linewidth=0.8, zorder=0)
            sep_y += 0.3
        for sk in fam:
            if sk in ERROR_SOLVERS:
                sep_y += 0.7

    ax.set_xscale("log")
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.4, zorder=0)
    ax.invert_yaxis()

legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ROBOT_COLORS["panda"],
           markersize=9, label=ROBOT_LABELS["panda"]),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ROBOT_COLORS["fetch"],
           markersize=9, label=ROBOT_LABELS["fetch"]),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ROBOT_COLORS["baxter"],
           markersize=9, label=ROBOT_LABELS["baxter"]),
]
fig.legend(handles=legend_els, loc="lower center", ncol=3,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Batch IK Median Error: No Collision (■) vs Collision-Free (◆)",
             fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_DIR / "ik_batch_error_cf_vs_nocf.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "ik_batch_error_cf_vs_nocf.png", dpi=200, bbox_inches="tight")
print("Wrote ik_batch_error_cf_vs_nocf.pdf/.png")
plt.close(fig)


# ── Figure 2: CF success rate heatmap ─────────────────────────────────────────

print("\n── Figure 2: Collision-free success-rate heatmap (batch, 256 problems) ──")

sub_cf = df[(df["mode"] == "batch") & (df["collision_free"] == True)]

CF_PLOT_ORDER = ["HJCD-JAX", "HJCD-CUDA", "LS-JAX", "LS-CUDA",
                 "SQP-JAX",  "SQP-CUDA",  "MPPI-JAX", "MPPI-CUDA",
                 "PyRoki", "cuRobo"]

heat = pd.DataFrame(index=CF_PLOT_ORDER, columns=ROBOTS, dtype=float)
for sk in CF_PLOT_ORDER:
    for robot in ROBOTS:
        row = sub_cf[(sub_cf["solver_key"] == sk) & (sub_cf["robot"] == robot)]
        if not row.empty:
            conv_rate = float(row["success_rate"].values[0])
            cf_n = row["coll_free_n"].values[0]
            total = row["success_total"].values[0]
            cf_rate = float(cf_n) / float(total) * 100 if pd.notna(cf_n) and float(total) > 0 else conv_rate
            heat.loc[sk, robot] = min(conv_rate, cf_rate)

fig, ax = plt.subplots(figsize=(5, 5.5))
im = ax.imshow(heat.values.astype(float), vmin=0, vmax=100,
               cmap="RdYlGn", aspect="auto")
plt.colorbar(im, ax=ax, label="Success rate (%)")

ax.set_xticks(range(len(ROBOTS)))
ax.set_xticklabels([ROBOT_LABELS[r] for r in ROBOTS], rotation=20, ha="right", fontsize=9)
ax.set_yticks(range(len(CF_PLOT_ORDER)))
ax.set_yticklabels([SOLVER_DISPLAY.get(s, s) for s in CF_PLOT_ORDER], fontsize=9)
ax.set_title("Collision-Free IK Success Rate\n(Batch, 256 problems)", fontsize=10)

for i, sk in enumerate(CF_PLOT_ORDER):
    for j, robot in enumerate(ROBOTS):
        v = heat.loc[sk, robot]
        if not pd.isna(v):
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if 20 < v < 80 else "white")

fig.tight_layout()
fig.savefig(OUT_DIR / "ik_cf_success_heatmap.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "ik_cf_success_heatmap.png", dpi=200, bbox_inches="tight")
print("Wrote ik_cf_success_heatmap.pdf/.png")
plt.close(fig)

# ── Figure: Combined CF + no-CF bar chart (grouped by algorithm) ──────────────

print("\n── Figure: Combined CF vs no-CF latency bar charts ──")

def _combined_latency_fig(df_all, mode, title, out_stem, my_groups=None, vertical_legend=False):
    """
    Single grouped bar chart: major x-groups = algorithm family,
    minor x-groups = robot.  Each robot gets 4 bars (JAX/CUDA × CF/no-CF)
    for 'my' methods and 2 bars for baselines.  Hatching distinguishes robots.
    my_groups: override the default list of (label, jsk, csk) tuples.
    """
    n_probs = 256 if mode == "batch" else 1

    _DEFAULT_MY_GROUPS = [
        ("PyRoNot-HJCD", "HJCD-JAX", "HJCD-CUDA"),
        ("PyRoNot-LS",   "LS-JAX",   "LS-CUDA"),
        ("PyRoNot-SQP",  "SQP-JAX",  "SQP-CUDA"),
        ("PyRoNot-MPPI", "MPPI-JAX", "MPPI-CUDA"),
    ]
    MY_GROUPS = my_groups if my_groups is not None else _DEFAULT_MY_GROUPS
    OTHER_GROUPS = [
        ("cuRobo", "CuRobo"),
        ("PyRoki", "PyRoKi"),
    ]

    C_JAX_NOCF    = "#CC2288"
    C_JAX_CF      = "#E88ABF"
    C_CUDA_NOCF   = "#DD5544"
    C_CUDA_CF     = "#F2A899"
    C_CUROBO_NOCF = "#55A868"
    C_CUROBO_CF   = "#A5D6AF"
    C_PYROKI_NOCF = "#4C72B0"
    C_PYROKI_CF   = "#9EB8D9"

    OTHER_COLORS = {
        "CuRobo": (C_CUROBO_NOCF, C_CUROBO_CF),
        "PyRoKi": (C_PYROKI_NOCF, C_PYROKI_CF),
    }

    ROBOT_HATCH = {"panda": "", "fetch": "///", "baxter": "xxx"}
    ROBOT_SHORT_LABEL = {"panda": "Panda", "fetch": "Fetch", "baxter": "Baxter"}

    sub_nocf = df_all[(df_all["mode"] == mode) & (df_all["collision_free"] == False)]
    sub_cf   = df_all[(df_all["mode"] == mode) & (df_all["collision_free"] == True)]

    bw         = 0.4   # single bar width
    robot_gap  = 0.14   # gap between robot sub-groups within an algo group
    algo_gap   = 0.80   # gap between algorithm groups

    n_groups = len(MY_GROUPS) + len(OTHER_GROUPS)
    fig_w = max(10, n_groups * 4.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 8))

    def get_t(rdf, sk):
        if sk not in rdf.index:
            return np.nan
        t = rdf.loc[sk, "t_med_ms"]
        return float(t) * 256 if not pd.isna(t) else np.nan

    cx = 0.0
    robot_tick_pos    = []
    robot_tick_labels = []
    algo_group_spans  = []   # (x_lo, x_hi, label)

    for label, jsk, csk in MY_GROUPS:
        group_x_lo = cx
        for robot in ROBOTS:
            rdf_nocf = sub_nocf[sub_nocf["robot"] == robot].set_index("solver_key")
            rdf_cf   = sub_cf  [sub_cf  ["robot"] == robot].set_index("solver_key")

            vals   = [get_t(rdf_nocf, jsk), get_t(rdf_cf, jsk),
                      get_t(rdf_nocf, csk), get_t(rdf_cf, csk)]
            colors = [C_JAX_NOCF, C_JAX_CF, C_CUDA_NOCF, C_CUDA_CF]
            hatch  = ROBOT_HATCH[robot]

            for i, (val, col) in enumerate(zip(vals, colors)):
                if not np.isnan(val):
                    ax.bar(cx + i * bw, val, width=bw * 0.88, color=col,
                           hatch=hatch, edgecolor="white", linewidth=0.5, zorder=2)

            robot_tick_pos.append(cx + 1.5 * bw)
            robot_tick_labels.append(ROBOT_SHORT_LABEL[robot])
            cx += 4 * bw + robot_gap

        algo_group_spans.append((group_x_lo, cx - robot_gap, label))
        cx += algo_gap

    for label, sk in OTHER_GROUPS:
        group_x_lo = cx
        for robot in ROBOTS:
            rdf_nocf = sub_nocf[sub_nocf["robot"] == robot].set_index("solver_key")
            rdf_cf   = sub_cf  [sub_cf  ["robot"] == robot].set_index("solver_key")

            vals   = [get_t(rdf_nocf, sk), get_t(rdf_cf, sk)]
            c_nocf, c_cf = OTHER_COLORS[sk]
            hatch  = ROBOT_HATCH[robot]

            for i, (val, col) in enumerate(zip(vals, [c_nocf, c_cf])):
                if not np.isnan(val):
                    ax.bar(cx + i * bw, val, width=bw * 0.88, color=col,
                           hatch=hatch, edgecolor="white", linewidth=0.5, zorder=2)

            # Centre tick in a padded block matching MY_GROUPS width so labels
            # have the same horizontal space (pad with empty bars on each side)
            robot_tick_pos.append(cx + 1.5 * bw)
            robot_tick_labels.append(ROBOT_SHORT_LABEL[robot])
            cx += 4 * bw + robot_gap   # same block width as MY_GROUPS

        algo_group_spans.append((group_x_lo, cx - robot_gap, label))
        cx += algo_gap

    # Primary x-ticks: robot labels
    ax.set_xticks(robot_tick_pos)
    ax.set_xticklabels(robot_tick_labels, fontsize=15, rotation=30, ha="right")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Median time (ms)", fontsize=20)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.4, zorder=0)
    ax.set_xlim(-0.4, cx - algo_gap + 0.4)

    # Algo-group labels below robot ticks, separator lines between groups
    y_anno = -0.17   # axes-fraction y position for group labels
    for i, (x_lo, x_hi, lbl) in enumerate(algo_group_spans):
        xmid = (x_lo + x_hi) / 2
        ax.annotate(lbl,
                    xy=(xmid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -52), textcoords="offset points",
                    ha="center", va="top", fontsize=20, fontweight="bold",
                    annotation_clip=False)
        if i > 0:
            sep_x = (algo_group_spans[i-1][1] + x_lo) / 2
            ax.axvline(sep_x, color="gray", lw=1.0, ls="--", alpha=0.45, zorder=1)

    # Legend: condition colours + robot hatching
    if vertical_legend:
        condition_patches = [
            Patch(facecolor=C_JAX_NOCF,    label="JAX, No-CF"),
            Patch(facecolor=C_JAX_CF,      label="JAX, CF"),
            Patch(facecolor=C_CUDA_NOCF,   label="CUDA, No-CF"),
            Patch(facecolor=C_CUDA_CF,     label="CUDA, CF"),
            Patch(facecolor=C_CUROBO_NOCF, label="cuRobo, No-CF"),
            Patch(facecolor=C_CUROBO_CF,   label="cuRobo, CF"),
            Patch(facecolor=C_PYROKI_NOCF, label="PyRoki, No-CF"),
            Patch(facecolor=C_PYROKI_CF,   label="PyRoki, CF"),
        ]
        robot_patches = [
            Patch(facecolor="#aaaaaa", hatch="",    edgecolor="white", label="Panda (7-DOF)"),
            Patch(facecolor="#aaaaaa", hatch="///", edgecolor="white", label="Fetch (8-DOF)"),
            Patch(facecolor="#aaaaaa", hatch="xxx", edgecolor="white", label="Baxter (15-DOF)"),
        ]
        ax.legend(handles=condition_patches + robot_patches,
                  loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=True, fontsize=14, ncol=1,
                  title="Legend", title_fontsize=14)
        fig.suptitle(title, fontsize=22)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    else:
        condition_patches = [
            Patch(facecolor=C_JAX_NOCF,    label="JAX — No Collision"),
            Patch(facecolor=C_JAX_CF,      label="JAX — Collision Free"),
            Patch(facecolor=C_CUDA_NOCF,   label="CUDA — No Collision"),
            Patch(facecolor=C_CUDA_CF,     label="CUDA — Collision Free"),
            Patch(facecolor=C_CUROBO_NOCF, label="cuRobo — No Collision"),
            Patch(facecolor=C_CUROBO_CF,   label="cuRobo — Collision Free"),
            Patch(facecolor=C_PYROKI_NOCF, label="PyRoki — No Collision"),
            Patch(facecolor=C_PYROKI_CF,   label="PyRoki — Collision Free"),
        ]
        robot_patches = [
            Patch(facecolor="#aaaaaa", hatch="",    edgecolor="white", label="Panda (7-DOF)"),
            Patch(facecolor="#aaaaaa", hatch="///", edgecolor="white", label="Fetch (8-DOF)"),
            Patch(facecolor="#aaaaaa", hatch="xxx", edgecolor="white", label="Baxter (15-DOF)"),
        ]
        fig.legend(handles=condition_patches + robot_patches,
                   loc="upper center", ncol=6, frameon=False,
                   fontsize=15, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(title, fontsize=22, y=1.09)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out_stem}.pdf/.png")
    plt.close(fig)


_combined_latency_fig(
    df, mode="sequential",
    title="Sequential IK Latency — CF vs No-Collision",
    out_stem="ik_latency_seq_combined_cf",
)

_combined_latency_fig(
    df, mode="batch",
    title="Batch IK Latency (256 Problems) — CF vs No-Collision",
    out_stem="ik_latency_batch_combined_cf",
)

_combined_latency_fig(
    df, mode="batch",
    title="Batch IK Latency (256 Problems)",
    out_stem="ik_latency_batch_mppi_baselines_cf",
    my_groups=[("PyRoNot-MPPI", "MPPI-JAX", "MPPI-CUDA")],
    vertical_legend=True,
)

print("\nDone.")
