from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_header(value):
    """
    Standardize a column header so differences in capitalization,
    punctuation, or extra spaces do not prevent matching.
    """
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def find_column(dataframe, possible_names):
    """
    Find a dataframe column using one or more possible header names.
    """
    normalized_columns = {
        normalize_header(column): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        normalized_name = normalize_header(possible_name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    available = "\n".join(
        f"  - {column}"
        for column in dataframe.columns
    )

    raise KeyError(
        "Could not locate the required column.\n\n"
        "Names searched:\n"
        + "\n".join(f"  - {name}" for name in possible_names)
        + "\n\nAvailable columns:\n"
        + available
    )


def extract_numeric_rat_id(series):
    """
    Convert IDs such as 5, 5.0, or 'RAT 5' into the number 5.
    """
    return pd.to_numeric(
        series.astype(str).str.extract(
            r"(\d+)",
            expand=False
        ),
        errors="coerce"
    )


def create_violin_plot(
    dataframe,
    rating_column,
    x_axis_label,
    figure_title,
    output_stem
):
    """
    Create a violin plot with individual observations,
    median, IQR, sample sizes, and Spearman statistics.
    """

    clean = dataframe[
        ["Rat_ID", rating_column, "Partial_Percent"]
    ].dropna().copy()

    clean = clean.sort_values(
        by=[rating_column, "Rat_ID"]
    )

    if clean.empty:
        raise ValueError(
            f"No complete observations were found for {rating_column}."
        )

    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    rho, p_value = spearmanr(
        clean[rating_column],
        clean["Partial_Percent"]
    )

    sample_size = len(clean)

    print("\n" + figure_title)
    print("-" * len(figure_title))
    print(f"Matched observations: {sample_size}")
    print(f"Spearman rho: {rho:.6f}")
    print(f"Two-sided p-value: {p_value:.6f}")

    # --------------------------------------------------------
    # Prepare rating groups
    # --------------------------------------------------------

    ratings = sorted(
        clean[rating_column].unique()
    )

    groups = [
        clean.loc[
            clean[rating_column] == rating,
            "Partial_Percent"
        ].to_numpy()
        for rating in ratings
    ]

    positions = np.arange(
        1,
        len(ratings) + 1
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9.4, 6.4)
    )

    # Leave room on the right for the statistical annotation.
    fig.subplots_adjust(
        left=0.11,
        right=0.78,
        bottom=0.17,
        top=0.82
    )

    # --------------------------------------------------------
    # Draw violin distributions
    # --------------------------------------------------------

    violin_groups = []
    violin_positions = []

    for position, values in zip(
        positions,
        groups
    ):
        # Do not estimate a density for one observation or
        # for a category where all values are identical.
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

    # --------------------------------------------------------
    # Add observations, median, and IQR
    # --------------------------------------------------------

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

        # Small caps indicating Q1 and Q3.
        ax.hlines(
            [q1, q3],
            position - 0.065,
            position + 0.065,
            color="#333333",
            linewidth=1.4,
            zorder=2
        )

        # Add horizontal jitter only.
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

        # Median marker.
        ax.scatter(
            position,
            median,
            s=44,
            facecolor="white",
            edgecolor="#111111",
            linewidth=1.1,
            zorder=5
        )

    # --------------------------------------------------------
    # Axis range and sample sizes
    # --------------------------------------------------------

    partial_values = clean[
        "Partial_Percent"
    ].to_numpy()

    data_min = np.min(partial_values)
    data_max = np.max(partial_values)
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

    label_height = (
        data_max + 0.10 * data_range
    )

    for position, values in zip(
        positions,
        groups
    ):
        ax.text(
            position,
            label_height,
            f"n = {len(values)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222"
        )

    # --------------------------------------------------------
    # Statistical annotation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Labels and appearance
    # --------------------------------------------------------

    ax.set_xticks(positions)

    ax.set_xticklabels(
        [f"{rating:g}" for rating in ratings]
    )

    ax.set_xlabel(
        x_axis_label,
        fontsize=11
    )

    ax.set_ylabel(
        "Erroneous partial detections (%)",
        fontsize=11
    )

    fig.suptitle(
        figure_title,
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

    # --------------------------------------------------------
    # Save PNG and PDF
    # --------------------------------------------------------

    output_png = workbook.with_name(
        f"{output_stem}.png"
    )

    output_pdf = workbook.with_name(
        f"{output_stem}.pdf"
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight"
    )

    print(f"PNG saved to: {output_png}")
    print(f"PDF saved to: {output_pdf}")

    plt.show()


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
# READ THE REQUIRED SHEETS
#
# header=1 means Excel row 2 contains the column names.
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

    print("\nAvailable worksheets:")
    for sheet_name in excel_file.sheet_names:
        print(f"  - {sheet_name}")

    raise ValueError(
        "The workbook must contain sheets named "
        "'Quality Ratings' and 'Data'."
    ) from error


# ============================================================
# FIND THE REQUIRED COLUMNS BY HEADER NAME
# ============================================================

quality_id_column = find_column(
    quality,
    [
        "ID #",
        "ID",
        "Rat ID"
    ]
)

average_shape_column = find_column(
    quality,
    [
        "Average Shape accuracy",
        "Average Shape Accuracy",
        "Average shape-accuracy rating"
    ]
)

average_visibility_column = find_column(
    quality,
    [
        "Average Visibility of detection",
        "Average Visibility of Detection",
        "Average visibility rating"
    ]
)

data_id_column = find_column(
    data,
    [
        "Rat ID",
        "ID #",
        "ID"
    ]
)

partial_count_column = find_column(
    data,
    [
        "Partial detection",
        "Partial Detection",
        "Partial detection count"
    ]
)

total_count_column = find_column(
    data,
    [
        "Total",
        "Total detections",
        "Total Detections"
    ]
)


# ============================================================
# BUILD THE RATING TABLE
# ============================================================

ratings_df = pd.DataFrame({
    "Rat_ID": extract_numeric_rat_id(
        quality[quality_id_column]
    ),
    "Average_Shape_Accuracy": pd.to_numeric(
        quality[average_shape_column],
        errors="coerce"
    ),
    "Average_Visibility": pd.to_numeric(
        quality[average_visibility_column],
        errors="coerce"
    )
})


# ============================================================
# BUILD THE PARTIAL-DETECTION TABLE
#
# Partial detection percentage is calculated from the raw
# counts so the script does not depend on cached Excel formulas.
# ============================================================

partial_df = pd.DataFrame({
    "Rat_ID": extract_numeric_rat_id(
        data[data_id_column]
    ),
    "Partial_Count": pd.to_numeric(
        data[partial_count_column],
        errors="coerce"
    ),
    "Total_Detections": pd.to_numeric(
        data[total_count_column],
        errors="coerce"
    )
})

partial_df.loc[
    partial_df["Total_Detections"] <= 0,
    "Total_Detections"
] = np.nan

partial_df["Partial_Percent"] = (
    partial_df["Partial_Count"]
    / partial_df["Total_Detections"]
    * 100
)


# ============================================================
# MERGE THE SHEETS BY RAT ID
# ============================================================

combined_df = ratings_df.merge(
    partial_df[
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

print(
    f"\nMatched rat IDs before removing missing values: "
    f"{len(combined_df)}"
)


# ============================================================
# CREATE BOTH FIGURES
# ============================================================

create_violin_plot(
    dataframe=combined_df,
    rating_column="Average_Shape_Accuracy",
    x_axis_label=(
        "Average shape-accuracy rating\n"
        "(higher values indicate greater perceived accuracy)"
    ),
    figure_title=(
        "Association between shape-accuracy ratings\n"
        "and partial-detection error"
    ),
    output_stem=(
        "Shape_Accuracy_vs_Partial_Detection_Violin"
    )
)

create_violin_plot(
    dataframe=combined_df,
    rating_column="Average_Visibility",
    x_axis_label=(
        "Average visibility rating\n"
        "(higher values indicate greater perceived visibility)"
    ),
    figure_title=(
        "Association between visibility ratings\n"
        "and partial-detection error"
    ),
    output_stem=(
        "Visibility_vs_Partial_Detection_Violin"
    )
)
