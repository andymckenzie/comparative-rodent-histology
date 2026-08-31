from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# SELECT THE WORKBOOK
# ============================================================

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

selected_file = askopenfilename(
    title="Select the Master Rat Excel workbook",
    filetypes=[
        ("Excel workbooks", "*.xlsx"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not selected_file:
    raise SystemExit("No workbook was selected.")

workbook = Path(selected_file)

print(f"Workbook selected: {workbook}")


# ============================================================
# OUTPUT FILES
# ============================================================

output_png = workbook.with_name(
    "Shape_Accuracy_vs_Partial_Detection_Violin.png"
)

output_pdf = workbook.with_name(
    "Shape_Accuracy_vs_Partial_Detection_Violin.pdf"
)


# ============================================================
# READ WORKSHEETS
#
# Excel row 2 is treated as the header row.
# ============================================================

try:
    quality = pd.read_excel(
        workbook,
        sheet_name="Quality Ratings",
        header=1,
        engine="openpyxl"
    )

    data = pd.read_excel(
        workbook,
        sheet_name="Data",
        header=1,
        engine="openpyxl"
    )

except ValueError as error:
    excel_file = pd.ExcelFile(
        workbook,
        engine="openpyxl"
    )

    print("\nAvailable worksheet names:")
    for sheet_name in excel_file.sheet_names:
        print(f"  - {sheet_name}")

    raise ValueError(
        "\nCould not find one of the required worksheets.\n"
        "The script requires sheets named exactly:\n"
        "  Quality Ratings\n"
        "  Data"
    ) from error


# ============================================================
# EXTRACT SHAPE-ACCURACY RATINGS
#
# Quality Ratings:
# Column A = Rat ID
# Column B = Sarah shape accuracy
# Column D = Andria shape accuracy
# ============================================================

if quality.shape[1] < 4:
    raise ValueError(
        "The Quality Ratings sheet does not contain enough columns."
    )

quality_clean = pd.DataFrame({
    "Rat_ID": pd.to_numeric(
        quality.iloc[:, 0],
        errors="coerce"
    ),
    "Sarah_Shape": pd.to_numeric(
        quality.iloc[:, 1],
        errors="coerce"
    ),
    "Andria_Shape": pd.to_numeric(
        quality.iloc[:, 3],
        errors="coerce"
    )
})

quality_clean["Shape_Accuracy"] = quality_clean[
    ["Sarah_Shape", "Andria_Shape"]
].mean(
    axis=1,
    skipna=False
)


# ============================================================
# EXTRACT PARTIAL-DETECTION DATA
#
# Data:
# Column A = Rat ID
# Column D = Partial-detection count
# Column G = Total detections
# ============================================================

if data.shape[1] < 7:
    raise ValueError(
        "The Data sheet does not contain enough columns."
    )

data_clean = pd.DataFrame({
    "Rat_ID_Text": data.iloc[:, 0].astype(str),
    "Partial_Count": pd.to_numeric(
        data.iloc[:, 3],
        errors="coerce"
    ),
    "Total_Detections": pd.to_numeric(
        data.iloc[:, 6],
        errors="coerce"
    )
})

data_clean["Rat_ID"] = pd.to_numeric(
    data_clean["Rat_ID_Text"].str.extract(
        r"(\d+)",
        expand=False
    ),
    errors="coerce"
)

data_clean.loc[
    data_clean["Total_Detections"] <= 0,
    "Total_Detections"
] = np.nan

data_clean["Partial_Percent"] = (
    data_clean["Partial_Count"]
    / data_clean["Total_Detections"]
    * 100
)


# ============================================================
# MERGE DATA BY RAT ID
# ============================================================

plot_df = quality_clean.merge(
    data_clean[
        [
            "Rat_ID",
            "Partial_Count",
            "Total_Detections",
            "Partial_Percent"
        ]
    ],
    on="Rat_ID",
    how="inner"
)

plot_df = plot_df.dropna(
    subset=[
        "Rat_ID",
        "Shape_Accuracy",
        "Partial_Percent"
    ]
).copy()

plot_df = plot_df.sort_values(
    by=["Shape_Accuracy", "Rat_ID"]
)

if plot_df.empty:
    raise ValueError(
        "No valid matched observations were found.\n"
        "Check the rat IDs and worksheet layouts."
    )


# ============================================================
# CALCULATE SPEARMAN CORRELATION
# ============================================================

rho, p_value = spearmanr(
    plot_df["Shape_Accuracy"],
    plot_df["Partial_Percent"],
    nan_policy="omit"
)

sample_size = len(plot_df)

print("\nSpearman correlation results")
print("--------------------------------")
print(f"Number of matched observations: {sample_size}")
print(f"Spearman rho: {rho:.6f}")
print(f"Two-sided p-value: {p_value:.6f}")


# ============================================================
# PREPARE RATING GROUPS
# ============================================================

ratings = sorted(
    plot_df["Shape_Accuracy"].unique()
)

groups = [
    plot_df.loc[
        plot_df["Shape_Accuracy"] == rating,
        "Partial_Percent"
    ].to_numpy()
    for rating in ratings
]

positions = np.arange(
    1,
    len(ratings) + 1
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.4, 6.4)
)

fig.subplots_adjust(
    left=0.11,
    right=0.78,
    bottom=0.16,
    top=0.82
)


# ============================================================
# DRAW VIOLINS
# ============================================================

violin_groups = []
violin_positions = []

for position, values in zip(
    positions,
    groups
):
    if len(values) >= 2 and np.ptp(values) > 0:
        violin_groups.append(values)
        violin_positions.append(position)

if violin_groups:
    violin_parts = ax.violinplot(
        violin_groups,
        positions=violin_positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott"
    )

    for body in violin_parts["bodies"]:
        body.set_facecolor("#D9E4EA")
        body.set_edgecolor("#667985")
        body.set_linewidth(0.9)
        body.set_alpha(0.85)


# ============================================================
# ADD OBSERVATIONS, IQR, AND MEDIAN
# ============================================================

random_generator = np.random.default_rng(
    seed=42
)

point_colors = plt.cm.tab10(
    np.linspace(0, 1, len(ratings))
)

for position, values, point_color in zip(
    positions,
    groups,
    point_colors
):
    median = np.median(values)

    q1 = np.percentile(
        values,
        25,
        method="linear"
    )

    q3 = np.percentile(
        values,
        75,
        method="linear"
    )

    # Thin IQR line behind the observations.
    ax.vlines(
        position,
        q1,
        q3,
        color="#333333",
        linewidth=1.4,
        zorder=2
    )

    # Small caps marking Q1 and Q3.
    ax.hlines(
        [q1, q3],
        position - 0.065,
        position + 0.065,
        color="#333333",
        linewidth=1.4,
        zorder=2
    )

    # Slight horizontal jitter prevents overlapping points.
    jittered_x = random_generator.normal(
        loc=position,
        scale=0.045,
        size=len(values)
    )

    ax.scatter(
        jittered_x,
        values,
        s=40,
        facecolor=point_color,
        edgecolor="#333333",
        linewidth=0.55,
        alpha=0.82,
        zorder=3
    )

    # Median marker placed above the points.
    ax.scatter(
        position,
        median,
        s=44,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.1,
        zorder=5
    )


# ============================================================
# AXIS RANGE AND SAMPLE-SIZE LABELS
# ============================================================

all_partial_values = plot_df[
    "Partial_Percent"
].to_numpy()

data_min = np.min(all_partial_values)
data_max = np.max(all_partial_values)
data_range = data_max - data_min

if data_range == 0:
    data_range = 1

lower_limit = max(
    0,
    data_min - 0.08 * data_range
)

upper_limit = (
    data_max + 0.20 * data_range
)

ax.set_ylim(
    lower_limit,
    upper_limit
)

n_label_height = (
    data_max + 0.10 * data_range
)

for position, values in zip(
    positions,
    groups
):
    ax.text(
        position,
        n_label_height,
        f"n = {len(values)}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#222222"
    )


# ============================================================
# SPEARMAN STATISTICS BOX
# ============================================================

if p_value < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {p_value:.3f}"

statistics_text = (
    f"Spearman ρ = {rho:.3f}\n"
    f"{p_text}\n"
    f"n = {sample_size}"
)

fig.text(
    0.805,
    0.77,
    statistics_text,
    ha="left",
    va="top",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.45",
        "facecolor": "white",
        "edgecolor": "#333333",
        "linewidth": 0.9
    }
)


# ============================================================
# LABELS AND FORMATTING
# ============================================================

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    [
        f"{rating:g}"
        for rating in ratings
    ]
)

ax.set_xlabel(
    "Average shape-accuracy rating\n"
    "(higher values indicate greater perceived accuracy)",
    fontsize=11
)

ax.set_ylabel(
    "Erroneous partial detections (%)",
    fontsize=11
)

fig.suptitle(
    "Association between shape-accuracy ratings\n"
    "and partial-detection error",
    fontsize=14,
    fontweight="bold",
    y=0.96
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.65,
    alpha=0.4,
    zorder=0
)


# ============================================================
# SAVE FIGURE
# ============================================================

fig.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    output_pdf,
    bbox_inches="tight"
)

print("\nFigure files saved:")
print(f"PNG: {output_png}")
print(f"PDF: {output_pdf}")

plt.show()
