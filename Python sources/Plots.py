# -*- coding: utf-8 -*-
"""
Generate synthetic-drift summary plots directly from CSV files.

Required input files placed in the same directory as this script:
    synthetic_summary_gradual.csv
    synthetic_summary_abrupt.csv

Expected CSV columns:
    DriftType, Metric, Drifts, Features, Scenario, Detector, Mean, Std

The program creates the same type of plots and summary files as the previous
manual-dictionary version, but all numerical values are read from the CSV files.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GRADUAL_CSV = BASE_DIR / "synthetic_summary_gradual.csv"
ABRUPT_CSV = BASE_DIR / "synthetic_summary_abrupt.csv"

OUT_DIR = BASE_DIR / "HAST_FeatureKS_result_plots_old_style_wins_fixed_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = BASE_DIR / "HAST_FeatureKS_result_plots_old_style_wins_fixed_labels.zip"

PROPOSED_DETECTOR = "HAST"
FEATURE_KS_DETECTOR = "Feature-KS"

# Required order on plots and tables.
detectors = [PROPOSED_DETECTOR, "ADWIN", "EDDM", "DDM", "OCDD", FEATURE_KS_DETECTOR]
other_detectors = [det for det in detectors if det != PROPOSED_DETECTOR]

metrics = ["D1", "D2", "R"]
drift_types = ["abrupt", "gradual"]
drifts = [3, 5, 10, 15]
features = [30, 60, 90]


# ============================================================
# FONT SETTINGS
# ============================================================
# These values are intentionally larger because several plots are later
# inserted as small subfigures in one row in the paper.

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
HEATMAP_TEXT_FONT_SIZE = 12
COLORBAR_LABEL_SIZE = 16
TABLE_FONT_SIZE = 11

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
})


# ============================================================
# DATA LOADING
# ============================================================

def normalize_detector_name(name):
    """Normalize possible spelling variants of Feature-KS."""
    if pd.isna(name):
        return name

    name = str(name).strip()

    aliases = {
        "Features-KS": "Feature-KS",
        "FeatureKS": "Feature-KS",
        "FeaturesKS": "Feature-KS",
        "FEATURE-KS": "Feature-KS",
        "Feature-FS": "Feature-KS",
        "Features-FS": "Feature-KS",
    }

    return aliases.get(name, name)


def read_summary_csv(path, expected_drift_type):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            f"Place {path.name} in the same directory as this script."
        )

    data = pd.read_csv(path)

    required_columns = {
        "DriftType", "Metric", "Drifts", "Features", "Scenario",
        "Detector", "Mean", "Std"
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"File {path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = data.copy()
    data["DriftType"] = data["DriftType"].astype(str).str.strip().str.lower()
    data["Metric"] = data["Metric"].astype(str).str.strip()
    data["Detector"] = data["Detector"].apply(normalize_detector_name)
    data["Drifts"] = pd.to_numeric(data["Drifts"], errors="raise").astype(int)
    data["Features"] = pd.to_numeric(data["Features"], errors="raise").astype(int)
    data["Mean"] = pd.to_numeric(data["Mean"], errors="raise")
    data["Std"] = pd.to_numeric(data["Std"], errors="coerce")

    if not (data["DriftType"] == expected_drift_type).all():
        found = sorted(data["DriftType"].unique())
        raise ValueError(
            f"File {path.name} should contain only DriftType='{expected_drift_type}', "
            f"but found: {found}"
        )

    return data


def validate_dataset(data):
    """Check whether all required detector/metric/scenario combinations exist."""
    keys = ["DriftType", "Metric", "Drifts", "Features", "Detector"]

    duplicated = data[data.duplicated(keys, keep=False)]
    if not duplicated.empty:
        duplicated.to_csv(OUT_DIR / "duplicated_rows_warning.csv", index=False)
        raise ValueError(
            "Duplicated rows found for some combinations. "
            f"Details saved to: {OUT_DIR / 'duplicated_rows_warning.csv'}"
        )

    present = set(tuple(row) for row in data[keys].itertuples(index=False, name=None))

    expected = set()
    for drift_type in drift_types:
        for metric in metrics:
            for n_drifts in drifts:
                for n_features in features:
                    for detector in detectors:
                        expected.add((drift_type, metric, n_drifts, n_features, detector))

    missing = sorted(expected.difference(present))

    if missing:
        missing_df = pd.DataFrame(
            missing,
            columns=["DriftType", "Metric", "Drifts", "Features", "Detector"]
        )
        missing_df.to_csv(OUT_DIR / "missing_required_rows.csv", index=False)
        raise ValueError(
            "Some required rows are missing. "
            f"Details saved to: {OUT_DIR / 'missing_required_rows.csv'}"
        )


def load_all_data():
    gradual = read_summary_csv(GRADUAL_CSV, "gradual")
    abrupt = read_summary_csv(ABRUPT_CSV, "abrupt")

    data = pd.concat([gradual, abrupt], ignore_index=True)

    # Keep only metrics, scenarios and detectors used in the paper plots.
    data = data[
        data["Metric"].isin(metrics)
        & data["DriftType"].isin(drift_types)
        & data["Drifts"].isin(drifts)
        & data["Features"].isin(features)
        & data["Detector"].isin(detectors)
    ].copy()

    validate_dataset(data)

    # Previous plotting code used the column name Value.
    data["Value"] = data["Mean"]

    # Stable categorical order in plots and summaries.
    data["Detector"] = pd.Categorical(data["Detector"], categories=detectors, ordered=True)
    data["Metric"] = pd.Categorical(data["Metric"], categories=metrics, ordered=True)
    data["DriftType"] = pd.Categorical(data["DriftType"], categories=drift_types, ordered=True)

    data = data.sort_values(["DriftType", "Metric", "Drifts", "Features", "Detector"])
    data.to_csv(OUT_DIR / "synthetic_means_loaded_from_csv.csv", index=False)

    return data


df = load_all_data()


# ============================================================
# HELPERS
# ============================================================

def save_fig(name):
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"

    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.20)
    plt.savefig(svg, bbox_inches="tight", pad_inches=0.20)
    plt.close()


def text_color_for_cell(value, vmin, vmax, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)

    if vmax == vmin:
        norm_value = 0.5
    else:
        norm_value = (value - vmin) / (vmax - vmin)

    r, g, b, _ = cmap(norm_value)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    return "white" if luminance < 0.5 else "black"


def metric_axis_label(metric):
    labels = {
        "D1": "D1: detection-centric distance (lower is better)",
        "D2": "D2: event-centric distance (lower is better)",
        "R": "R: alarm-count inconsistency (lower is better)",
    }
    return labels[metric]


def get_value(data, drift_type, metric, n_drifts, n_features, detector):
    value = data[
        (data["DriftType"] == drift_type)
        & (data["Metric"] == metric)
        & (data["Drifts"] == n_drifts)
        & (data["Features"] == n_features)
        & (data["Detector"] == detector)
    ]["Value"]

    if value.empty:
        raise ValueError(
            "Missing value for: "
            f"{drift_type}, {metric}, {n_drifts} drifts, "
            f"{n_features} features, detector={detector}"
        )

    return float(value.iloc[0])


# ============================================================
# SUMMARY
# ============================================================

rank_df = df.copy()
rank_df["Rank"] = rank_df.groupby(
    ["DriftType", "Metric", "Drifts", "Features"], observed=False
)["Value"].rank(method="min", ascending=True)

summary = (
    rank_df.groupby("Detector", observed=False)
    .agg(
        MeanRank=("Rank", "mean"),
        MedianRank=("Rank", "median"),
        MeanValue=("Value", "mean"),
    )
    .reindex(detectors)
    .reset_index()
)

wins = (
    rank_df[rank_df["Rank"] == 1]
    .groupby(["Detector", "Metric"], observed=False)
    .size()
    .unstack(fill_value=0)
    .reindex(detectors)
    .fillna(0)
)

for col in metrics:
    if col not in wins.columns:
        wins[col] = 0

wins = wins[metrics]
wins["Total"] = wins.sum(axis=1)

metric_means = (
    df.groupby(["Detector", "Metric"], observed=False)["Value"]
    .mean()
    .unstack()
    .reindex(detectors)
)

metric_stds = (
    df.groupby(["Detector", "Metric"], observed=False)["Std"]
    .mean()
    .unstack()
    .reindex(detectors)
)

metric_means.to_csv(OUT_DIR / "mean_metrics_by_detector.csv")
metric_stds.to_csv(OUT_DIR / "mean_std_by_detector.csv")
summary.to_csv(OUT_DIR / "mean_ranks_by_detector.csv", index=False)
wins.to_csv(OUT_DIR / "wins_by_metric.csv")


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_pareto(x_metric, y_metric, filename):
    x_df = df[df["Metric"] == x_metric].rename(columns={"Value": x_metric})
    y_df = df[df["Metric"] == y_metric].rename(columns={"Value": y_metric})

    data = pd.merge(
        x_df[["DriftType", "Drifts", "Features", "Detector", x_metric]],
        y_df[["DriftType", "Drifts", "Features", "Detector", y_metric]],
        on=["DriftType", "Drifts", "Features", "Detector"],
    )

    plt.figure(figsize=(8.5, 6))

    for det in detectors:
        for dtype, marker in [("gradual", "o"), ("abrupt", "s")]:
            sub = data[
                (data["Detector"] == det)
                & (data["DriftType"] == dtype)
            ]

            plt.scatter(
                sub[x_metric],
                sub[y_metric],
                marker=marker,
                label=f"{det} ({dtype})",
                alpha=0.75,
            )

    plt.xlabel(metric_axis_label(x_metric))
    plt.ylabel(metric_axis_label(y_metric))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE - 1, ncol=2)

    save_fig(filename)


def plot_mean_rank():
    ordered = summary.sort_values("MeanRank")

    plt.figure(figsize=(8.5, 5))
    plt.bar(ordered["Detector"].astype(str), ordered["MeanRank"])
    plt.ylabel("Mean rank across D1, D2 and R (lower is better)")
    plt.grid(axis="y", alpha=0.3)

    for i, v in enumerate(ordered["MeanRank"]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)

    save_fig("02_mean_rank_all_metrics")


def plot_wins_by_metric():
    wins_plot = wins[metrics].copy()

    plt.figure(figsize=(9, 5))

    bottom = np.zeros(len(wins_plot))
    x = np.arange(len(wins_plot.index))

    for metric in metrics:
        vals = wins_plot[metric].values
        plt.bar(x, vals, bottom=bottom, label=metric)
        bottom += vals

    plt.xticks(x, wins_plot.index.astype(str))
    plt.ylabel("Number of wins")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)

    for i, total in enumerate(bottom):
        plt.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)

    save_fig("03_scenario_wins_by_metric")


def plot_metric_heatmap(metric, drift_type, filename):
    row_labels = [f"{d}d {f}f" for d in drifts for f in features]

    mat = []

    for d in drifts:
        for f in features:
            row = []

            for det in detectors:
                row.append(get_value(df, drift_type, metric, d, f, det))

            mat.append(row)

    mat = np.array(mat, dtype=float)

    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))

    plt.figure(figsize=(8.3, 7.2))
    plt.imshow(mat, aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(f"{metric} score (lower is better)", fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    plt.xticks(np.arange(len(detectors)), detectors, rotation=20, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_TEXT_FONT_SIZE,
                color=text_color_for_cell(mat[i, j], vmin, vmax),
            )

    plt.xlabel("Detector")
    plt.ylabel("Scenario")

    save_fig(filename)



def build_heatmap_matrix(metric, drift_type, detector_list=None):
    """Return matrix and row labels for one metric and one drift type."""
    if detector_list is None:
        detector_list = detectors

    row_labels = [f"{d}d {f}f" for d in drifts for f in features]
    mat = []

    for d in drifts:
        for f in features:
            row = []
            for det in detector_list:
                row.append(get_value(df, drift_type, metric, d, f, det))
            mat.append(row)

    return np.array(mat, dtype=float), row_labels


def draw_heatmap_on_axis(ax, mat, row_labels, col_labels, title, show_y_labels=True):
    """Draw one heatmap on an existing axis and return the image object."""
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))

    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE + 1, pad=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=28, ha="right", fontsize=TICK_LABEL_SIZE - 2)
    ax.set_yticks(np.arange(len(row_labels)))

    if show_y_labels:
        ax.set_yticklabels(row_labels, fontsize=TICK_LABEL_SIZE - 2)
        ax.set_ylabel("Scenario", fontsize=AXIS_LABEL_SIZE - 1)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_TEXT_FONT_SIZE - 2,
                color=text_color_for_cell(mat[i, j], vmin, vmax),
            )

    return im


def plot_metric_heatmaps_two_panels(metric, filename):
    """Create one figure with abrupt and gradual heatmaps for one metric."""
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.4), sharey=True)

    for ax, drift_type, title in zip(
        axes,
        ["abrupt", "gradual"],
        ["Abrupt drift", "Gradual drift"],
    ):
        mat, row_labels = build_heatmap_matrix(metric, drift_type)
        im = draw_heatmap_on_axis(
            ax,
            mat,
            row_labels,
            detectors,
            f"{metric}: {title}",
            show_y_labels=(ax is axes[0]),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(f"{metric} score", fontsize=COLORBAR_LABEL_SIZE - 1)
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 2)

    fig.supxlabel("Detector", fontsize=AXIS_LABEL_SIZE, y=0.035)
    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.170,
        top=0.900,
        wspace=0.120,
    )
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def plot_all_metric_heatmaps_one_panel(filename):
    """Create one publication figure with six heatmaps: rows=drift type, columns=metric."""
    fig, axes = plt.subplots(2, 3, figsize=(24.0, 14.2), sharey=True)

    for row_idx, drift_type in enumerate(["abrupt", "gradual"]):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            mat, row_labels = build_heatmap_matrix(metric, drift_type)
            title = f"{metric}, {drift_type} drift"
            im = draw_heatmap_on_axis(
                ax,
                mat,
                row_labels,
                detectors,
                title,
                show_y_labels=(col_idx == 0),
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.020)
            cbar.set_label(f"{metric}", fontsize=COLORBAR_LABEL_SIZE - 2)
            cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 3)

    fig.supxlabel("Detector", fontsize=AXIS_LABEL_SIZE, y=0.035)
    fig.subplots_adjust(
        left=0.055,
        right=0.990,
        bottom=0.090,
        top=0.940,
        wspace=0.135,
        hspace=0.315,
    )
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def plot_proposed_advantage(metric, drift_type, filename):
    row_labels = [f"{d}d {f}f" for d in drifts for f in features]

    mat = []

    for d in drifts:
        for f in features:
            proposed_val = get_value(df, drift_type, metric, d, f, PROPOSED_DETECTOR)

            row = []

            for det in other_detectors:
                det_val = get_value(df, drift_type, metric, d, f, det)
                row.append(det_val - proposed_val)

            mat.append(row)

    mat = np.array(mat, dtype=float)

    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))

    plt.figure(figsize=(8.4, 7.2))
    plt.imshow(mat, aspect="auto")

    cbar = plt.colorbar()
    cbar.set_label(
        f"Delta {metric} = {metric}(detector) - {metric}({PROPOSED_DETECTOR})",
        fontsize=COLORBAR_LABEL_SIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)

    plt.xticks(np.arange(len(other_detectors)), other_detectors, rotation=20, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_TEXT_FONT_SIZE,
                color=text_color_for_cell(mat[i, j], vmin, vmax),
            )

    plt.xlabel("Compared detector")
    plt.ylabel("Scenario")

    save_fig(filename)


def plot_radar():
    means = metric_means[metrics].copy()
    scores = means.copy()

    for metric in metrics:
        mn = means[metric].min()
        mx = means[metric].max()

        if mx == mn:
            scores[metric] = 1.0
        else:
            scores[metric] = 1 - (means[metric] - mn) / (mx - mn)

    scores["Mean normalized score"] = scores[metrics].mean(axis=1)
    scores.to_csv(OUT_DIR / "radar_normalized_scores.csv")

    labels = [
        "D1 alarm localization",
        "D2 drift-event coverage",
        "R alarm consistency",
        "Mean normalized score",
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)

    for det in detectors:
        vals = scores.loc[
            det,
            ["D1", "D2", "R", "Mean normalized score"],
        ].tolist()

        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=det)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.38, 1.12),
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_radar_normalized_detector_profile.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "08_radar_normalized_detector_profile.svg", bbox_inches="tight")
    plt.close()


def plot_metric_vs_drifts(metric, filename):
    plot_df = (
        df[df["Metric"] == metric]
        .groupby(["Detector", "Drifts"], observed=False)["Value"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8.5, 5))

    for det in detectors:
        sub = plot_df[plot_df["Detector"] == det]

        plt.plot(
            sub["Drifts"],
            sub["Value"],
            marker="o",
            label=det,
        )

    plt.xlabel("Number of true drifts")
    plt.ylabel(f"Mean {metric} across features and drift regimes")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE, ncol=2)

    save_fig(filename)


def plot_compact_summary_table():
    compact = metric_means.copy()
    compact["Mean rank"] = summary.set_index("Detector").loc[compact.index, "MeanRank"]
    compact["Wins"] = wins.loc[compact.index, "Total"]
    compact = compact.round(3)

    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off")

    table = ax.table(
        cellText=compact.reset_index().values,
        colLabels=["Detector"] + list(compact.columns),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_compact_summary_table.png", dpi=600, bbox_inches="tight")
    plt.savefig(OUT_DIR / "11_compact_summary_table.svg", bbox_inches="tight")
    plt.close()


# ============================================================
# ADDITIONAL PLOTS: ABRUPT-ONLY AND GRADUAL-ONLY SUMMARIES
# ============================================================

def drift_type_summary_tables(drift_type):
    """Return mean ranks, wins and metric means for one drift type only."""
    sub = df[df["DriftType"] == drift_type].copy()

    sub["Rank"] = sub.groupby(
        ["Metric", "Drifts", "Features"], observed=False
    )["Value"].rank(method="min", ascending=True)

    summary_type = (
        sub.groupby("Detector", observed=False)
        .agg(
            MeanRank=("Rank", "mean"),
            MedianRank=("Rank", "median"),
            MeanValue=("Value", "mean"),
        )
        .reindex(detectors)
        .reset_index()
    )

    wins_type = (
        sub[sub["Rank"] == 1]
        .groupby(["Detector", "Metric"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(detectors)
        .fillna(0)
    )

    for col in metrics:
        if col not in wins_type.columns:
            wins_type[col] = 0

    wins_type = wins_type[metrics]
    wins_type["Total"] = wins_type.sum(axis=1)

    metric_means_type = (
        sub.groupby(["Detector", "Metric"], observed=False)["Value"]
        .mean()
        .unstack()
        .reindex(detectors)
    )

    return summary_type, wins_type, metric_means_type


def save_drift_type_csv(drift_type):
    summary_type, wins_type, metric_means_type = drift_type_summary_tables(drift_type)
    summary_type.to_csv(OUT_DIR / f"mean_ranks_by_detector_{drift_type}_only.csv", index=False)
    wins_type.to_csv(OUT_DIR / f"wins_by_metric_{drift_type}_only.csv")
    metric_means_type.to_csv(OUT_DIR / f"mean_metrics_by_detector_{drift_type}_only.csv")


def plot_mean_rank_by_drift_type(drift_type, filename):
    summary_type, _, _ = drift_type_summary_tables(drift_type)
    ordered = summary_type.sort_values("MeanRank")

    plt.figure(figsize=(8.5, 5))
    plt.bar(ordered["Detector"].astype(str), ordered["MeanRank"])
    plt.ylabel(f"Mean rank across D1, D2 and R, {drift_type} drift (lower is better)")
    plt.grid(axis="y", alpha=0.3)

    for i, v in enumerate(ordered["MeanRank"]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)

    save_fig(filename)


def plot_wins_by_metric_by_drift_type(drift_type, filename):
    _, wins_type, _ = drift_type_summary_tables(drift_type)
    wins_plot = wins_type[metrics].copy()

    plt.figure(figsize=(9, 5))

    bottom = np.zeros(len(wins_plot))
    x = np.arange(len(wins_plot.index))

    for metric in metrics:
        vals = wins_plot[metric].values
        plt.bar(x, vals, bottom=bottom, label=metric)
        bottom += vals

    plt.xticks(x, wins_plot.index.astype(str))
    plt.ylabel(f"Number of wins, {drift_type} drift")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)

    for i, total in enumerate(bottom):
        plt.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)

    save_fig(filename)


def plot_metric_vs_drifts_by_drift_type(metric, drift_type, filename):
    plot_df = (
        df[(df["Metric"] == metric) & (df["DriftType"] == drift_type)]
        .groupby(["Detector", "Drifts"], observed=False)["Value"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8.5, 5))

    for det in detectors:
        sub = plot_df[plot_df["Detector"] == det]
        plt.plot(sub["Drifts"], sub["Value"], marker="o", label=det)

    plt.xlabel("Number of true drifts")
    plt.ylabel(f"Mean {metric} across features, {drift_type} drift")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE, ncol=2)

    save_fig(filename)


def create_abrupt_gradual_extra_plots():
    for drift_type in ["abrupt", "gradual"]:
        save_drift_type_csv(drift_type)

        plot_mean_rank_by_drift_type(
            drift_type,
            f"19_mean_rank_{drift_type}_only",
        )

        plot_wins_by_metric_by_drift_type(
            drift_type,
            f"20_scenario_wins_by_metric_{drift_type}_only",
        )

        plot_metric_vs_drifts_by_drift_type(
            "D1",
            drift_type,
            f"21_D1_vs_number_of_drifts_{drift_type}_only",
        )

        plot_metric_vs_drifts_by_drift_type(
            "D2",
            drift_type,
            f"22_D2_vs_number_of_drifts_{drift_type}_only",
        )

        plot_metric_vs_drifts_by_drift_type(
            "R",
            drift_type,
            f"23_R_vs_number_of_drifts_{drift_type}_only",
        )


def _wins_for_panel(drift_type=None):
    if drift_type is None:
        return wins[metrics].copy()
    _, wins_type, _ = drift_type_summary_tables(drift_type)
    return wins_type[metrics].copy()


def plot_scenario_wins_three_panels(filename):
    """Create the three-panel wins plot with one shared y-axis label.

    Fixes:
    - only the left panel has the y-axis description,
    - detector labels are slightly rotated so OCDD and Feature-KS do not overlap,
    - the previous detector order is preserved, with Feature-KS appended.
    """
    panel_specs = [
        ("abrupt", "Abrupt drift"),
        ("gradual", "Gradual drift"),
        (None, "All scenarios"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 5.8), sharey=False)

    legend_handles = None
    legend_labels = None

    for panel_idx, (ax, (drift_type, panel_title)) in enumerate(zip(axes, panel_specs)):
        wins_plot = _wins_for_panel(drift_type)

        desired_order = [PROPOSED_DETECTOR, "ADWIN", "EDDM", "DDM", "OCDD", FEATURE_KS_DETECTOR]
        wins_plot = wins_plot.reindex([d for d in desired_order if d in wins_plot.index])

        bottom = np.zeros(len(wins_plot))
        x = np.arange(len(wins_plot.index))

        for metric in ["D1", "D2", "R"]:
            vals = wins_plot[metric].values
            ax.bar(x, vals, bottom=bottom, label=metric)
            bottom += vals

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_title(panel_title, fontsize=TITLE_FONT_SIZE + 2, pad=10)
        ax.set_xticks(x)

        # Slight rotation prevents the OCDD and Feature-KS labels from overlapping.
        ax.set_xticklabels(
            wins_plot.index.astype(str),
            fontsize=max(TICK_LABEL_SIZE - 2, 10),
            rotation=25,
            ha="right"
        )

        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax.grid(axis="y", alpha=0.30)
        ax.set_xlabel("")

        # Use only one y-axis label, on the left panel.
        if panel_idx == 0:
            ax.set_ylabel("Number of best-mean scenarios", fontsize=AXIS_LABEL_SIZE, labelpad=8)
        else:
            ax.set_ylabel("")

        for i, total in enumerate(bottom):
            ax.text(
                i,
                total + 0.25,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE + 1,
            )

        ax.set_ylim(0, max(bottom) + 4)

    axes[1].legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE + 1,
        frameon=True,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.23,
        top=0.77,
        wspace=0.18,
    )

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)



# ============================================================
# PREFERRED MULTI-PANEL HEATMAP LAYOUT
# ============================================================

def plot_all_metric_heatmaps_one_panel(filename):
    """Create the preferred compact 2 x 3 heatmap layout.

    Layout:
        columns = D1, D2, R
        rows    = abrupt drift, gradual drift

    Each metric column has one shared colorbar for the two drift types.
    The colorbars are placed in dedicated narrow columns, so they do not
    overlap the heatmaps or detector labels.
    """
    metric_titles = {
        "D1": r"$D_1$ score",
        "D2": r"$D_2$ score",
        "R": r"$R$ score",
    }
    row_titles = {
        "abrupt": "Abrupt drift",
        "gradual": "Gradual drift",
    }

    fig = plt.figure(figsize=(18.8, 8.2))
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.0, 0.035, 1.0, 0.035, 1.0, 0.035],
        wspace=0.24,
        hspace=0.30,
    )

    heatmap_axes = {}

    for col_idx, metric in enumerate(metrics):
        heatmap_col = 2 * col_idx
        cbar_col = heatmap_col + 1
        cax = fig.add_subplot(gs[:, cbar_col])

        mats = {}
        row_labels = None

        for drift_type in ["abrupt", "gradual"]:
            mat, labels = build_heatmap_matrix(metric, drift_type)
            mats[drift_type] = mat
            if row_labels is None:
                row_labels = labels

        combined = np.concatenate([mats["abrupt"].ravel(), mats["gradual"].ravel()])
        combined = combined[np.isfinite(combined)]

        if combined.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(combined))
            vmax = float(np.nanmax(combined))

        im_for_colorbar = None

        for row_idx, drift_type in enumerate(["abrupt", "gradual"]):
            ax = fig.add_subplot(gs[row_idx, heatmap_col])
            heatmap_axes[(row_idx, col_idx)] = ax
            mat = mats[drift_type]

            im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
            im_for_colorbar = im

            if row_idx == 0:
                ax.set_title(metric_titles.get(metric, f"{metric} score"), fontsize=TITLE_FONT_SIZE + 3, pad=9)

            ax.set_xticks(np.arange(len(detectors)))
            ax.set_xticklabels(detectors, rotation=28, ha="right", fontsize=TICK_LABEL_SIZE - 4)
            ax.set_yticks(np.arange(len(row_labels)))

            if col_idx == 0:
                ax.set_yticklabels(row_labels, fontsize=TICK_LABEL_SIZE - 4)
                ax.set_ylabel(row_titles[drift_type], fontsize=AXIS_LABEL_SIZE + 1, labelpad=27)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            ax.tick_params(axis="both", length=0)

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    value = mat[i, j]

                    if np.isfinite(value):
                        label = f"{value:.2f}"
                        color = text_color_for_cell(value, vmin, vmax)
                    else:
                        label = "--"
                        color = "black"

                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=HEATMAP_TEXT_FONT_SIZE - 3,
                        color=color,
                    )

        cbar = fig.colorbar(im_for_colorbar, cax=cax)
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 4)

    fig.supxlabel("Detector", fontsize=AXIS_LABEL_SIZE + 1, y=0.035)

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.165,
        top=0.910,
    )

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)

# ============================================================
# CREATE PLOTS
# ============================================================

plot_pareto("D2", "R", "01_pareto_D2_R_all_scenarios")
plot_mean_rank()
plot_wins_by_metric()

plot_metric_heatmap("R", "gradual", "04_R_heatmap_gradual")
plot_metric_heatmap("R", "abrupt", "05_R_heatmap_abrupt")
plot_metric_heatmap("D1", "gradual", "13_D1_heatmap_gradual")
plot_metric_heatmap("D1", "abrupt", "14_D1_heatmap_abrupt")

# Heatmaps collected into publication-style multi-panel figures.
plot_metric_heatmaps_two_panels("D1", "25_D1_heatmap_abrupt_gradual_one_panel")
plot_metric_heatmaps_two_panels("D2", "26_D2_heatmap_abrupt_gradual_one_panel")
plot_metric_heatmaps_two_panels("R", "27_R_heatmap_abrupt_gradual_one_panel")
plot_all_metric_heatmaps_one_panel("28_all_D1_D2_R_heatmaps_preferred_layout")

plot_proposed_advantage("R", "gradual", f"06_{PROPOSED_DETECTOR}_R_advantage_gradual")
plot_proposed_advantage("R", "abrupt", f"07_{PROPOSED_DETECTOR}_R_advantage_abrupt")
plot_proposed_advantage("D1", "gradual", f"15_{PROPOSED_DETECTOR}_D1_advantage_gradual")
plot_proposed_advantage("D1", "abrupt", f"16_{PROPOSED_DETECTOR}_D1_advantage_abrupt")

plot_radar()

plot_metric_vs_drifts("D2", "09_D2_vs_number_of_drifts")
plot_metric_vs_drifts("R", "10_R_vs_number_of_drifts")
plot_metric_vs_drifts("D1", "12_D1_vs_number_of_drifts")

plot_compact_summary_table()

plot_pareto("D1", "R", "17_pareto_D1_R_all_scenarios")
plot_pareto("D1", "D2", "18_pareto_D1_D2_all_scenarios")

create_abrupt_gradual_extra_plots()
plot_scenario_wins_three_panels("24_scenario_wins_three_panels_large_fonts")


# ============================================================
# ZIP PACKAGE
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in OUT_DIR.iterdir():
        if fn.is_file():
            zf.write(fn, arcname=fn.name)

print("=" * 80)
print("Finished.")
print(f"Input gradual CSV : {GRADUAL_CSV}")
print(f"Input abrupt CSV  : {ABRUPT_CSV}")
print(f"Output directory  : {OUT_DIR}")
print(f"ZIP file          : {ZIP_PATH}")
print("=" * 80)
