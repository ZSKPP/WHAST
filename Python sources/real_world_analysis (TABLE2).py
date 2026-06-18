# real_world_HAST_segment_stability_analysis.py
#
# Program objective:
# 1. Runs the same drift detectors as the previous program.
# 2. For HAST, treats alarms as boundaries between stream segments.
# 3. Checks whether HAST-defined segments are more stable than the entire stream.
# 4. Saves:
#    - global results of all methods,
#    - chunk-level error log,
#    - HAST segment-level metrics,
#    - analysis of error improvement before and after HAST alarms,
#    - plots illustrating segment stability.
# Main interpretation:
# If WithinSegmentStd < GlobalStd and the error decreases after HAST alarms,
# then it can be argued that HAST not only raises alarms, but also segments
# the stream into relatively stable periods.

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import hadamard
from scipy.io import arff
from scipy.stats import t, ks_2samp

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from river import drift
from river import naive_bayes
from river import tree


warnings.filterwarnings("ignore")


CONFIG = {
    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------
    # Set the input file here:
    #"input_file": "electricity.arff",
    #"input_file": "ozone.arff",
    #"input_file": "airlines.arff",
    #"input_file": "spam.arff",
    #"input_file": "covtype.arff",
    "input_file": "outdoor.arff",#
    #"input_file": "poker-hand.arff",
    #"input_file": "rialto.arff",
    #"input_file": "kddcup99.arff",
    #"input_file": "phishing.arff",
    #"input_file": "gas_sensor_full.arff",
    #"input_file": "INSECTS abrupt_balanced.csv",
    #"input_file": "INSECTS abrupt_imbalanced.csv",
    #"input_file": "INSECTS gradual_balanced.csv",
    #"input_file": "INSECTS gradual_imbalanced.csv",
    #"input_file": "INSECTS incremental_balanced.csv",
    "label_column": -1,

    # ------------------------------------------------------------
    # CLASSIFIERS & CHUNKS
    # ------------------------------------------------------------
    "classifier": "ht",       # "ht" albo "gnb"
    "chunk_size": 200,

    # ------------------------------------------------------------
    # ADAPTATION STRATEGY AFTER AN ALARM
    # ------------------------------------------------------------
    # Available:
    #"hard_reset"
    # "soft_retrain_recent"
    # "no_classifier_reset"
    "adaptation_strategy": "soft_retrain_recent",
    #"adaptation_strategy": "hard_reset",
    "adaptation_memory_chunks": 3,

    # ------------------------------------------------------------
    # HAST
    # ------------------------------------------------------------
    "HAST_alpha": 0.03,
    "HAST_theta": 0.25,
    "HAST_r": 12,
    "HAST_s": 50,
    "HAST_e": 20,
    "max_reference_chunks": 20,
    "HAST_cooldown": 15,

    # ------------------------------------------------------------
    # OTHER DETECTORS
    # ------------------------------------------------------------
    "adwin_delta": 0.002,

    "ocdd_nu": 0.05,
    "ocdd_size": 300,
    "ocdd_percent": 0.30,
    "ocdd_kernel": "rbf",
    "ocdd_gamma": "scale",

    # ------------------------------------------------------------
    # FEATURE-KS UNSUPERVISED BASELINE
    # ------------------------------------------------------------
    # Feature-wise two-sample Kolmogorov-Smirnov detector.
    # Drift is declared when the fraction of changed features
    # exceeds feature_ks_theta.
    "feature_ks_alpha": 0.03,
    "feature_ks_theta": 0.25,
    "feature_ks_max_reference_chunks": 20,
    "feature_ks_cooldown": 15,

    # ------------------------------------------------------------
    # SEGMENT STABILITY ANALYSIS
    # ------------------------------------------------------------
    # Number of chunks before and after an alarm to compare.
    # For short streams, set this value to 1 or 2.
    "before_after_window": 3,

    # # Minimum number of chunks in a segment required to compute its statistics.
    "min_segment_length": 2,

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------
    "results_dir": "results_segment_stability",
    "plots_dir": "plots_segment_stability",
}


def safe_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return (
        base.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("\\", "_")
    )


def load_dataset(file_path, label_column):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext == ".arff":
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

    else:
        raise ValueError("Only CSV and ARFF files are supported.")

    if label_column == -1:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    else:
        y = df[label_column]
        X = df.drop(columns=[label_column])

    y = pd.factorize(y)[0]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)

    return X.to_numpy(dtype=float), y.astype(int)


class HAST:
    def __init__(
        self,
        input_dim,
        chunk_size,
        alpha=0.05,
        theta=0.10,
        r=20,
        s=30,
        e=24,
        max_reference_chunks=20,
        cooldown=5,
        random_state=42,
    ):
        self.alpha = alpha
        self.theta = theta
        self.r = r
        self.s = s
        self.e = min(e, input_dim)

        self.chunk_size = chunk_size
        self.max_reference_chunks = max_reference_chunks
        self.max_reference_size = max_reference_chunks * chunk_size

        self.cooldown = cooldown
        self.cooldown_left = 0

        self.rng = np.random.default_rng(random_state)

        next_p2 = 1 << (input_dim - 1).bit_length()

        H = hadamard(next_p2).astype(float)
        H = H / np.sqrt(next_p2)

        self.W = H[:input_dim, :self.e]
        self.reference_memory = None

        self.t_crit = t.ppf(
            1.0 - self.alpha / 2.0,
            2 * self.s - 2
        )

        print(
            "HAST CONFIG:",
            f"alpha={self.alpha}",
            f"theta={self.theta}",
            f"r={self.r}",
            f"s={self.s}",
            f"e={self.e}",
            f"max_reference_chunks={self.max_reference_chunks}",
            f"cooldown={self.cooldown}",
            f"t_crit={self.t_crit:.4f}",
        )

    def relu(self, x):
        return np.maximum(0.0, x)

    def transform(self, X):
        return self.relu(np.dot(X, self.W))

    def detect(self, X):
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return False

        current = self.transform(X)

        if self.reference_memory is None:
            self.reference_memory = current
            return False

        s_eff = min(
            self.s,
            current.shape[0],
            self.reference_memory.shape[0],
        )

        if s_eff < 2:
            return False

        idx_current = self.rng.integers(
            0,
            current.shape[0],
            size=(self.r, s_eff),
        )

        idx_reference = self.rng.integers(
            0,
            self.reference_memory.shape[0],
            size=(self.r, s_eff),
        )

        current_samples = current[idx_current, :]
        reference_samples = self.reference_memory[idx_reference, :]

        mean_current = np.mean(current_samples, axis=1)
        mean_reference = np.mean(reference_samples, axis=1)

        var_current = np.var(current_samples, axis=1, ddof=1)
        var_reference = np.var(reference_samples, axis=1, ddof=1)

        pooled_var = (var_current + var_reference) / 2.0

        epsilon = 1e-12

        t_stat = np.abs(mean_current - mean_reference) / np.sqrt(
            pooled_var * (2.0 / s_eff) + epsilon
        )

        alarms = np.sum(t_stat > self.t_crit)
        alarm_threshold = self.theta * self.e * self.r

        drift_detected = alarms > alarm_threshold

        if drift_detected:
            self.reference_memory = current
            self.cooldown_left = self.cooldown
        else:
            self.reference_memory = np.vstack(
                (self.reference_memory, current)
            )

            if self.reference_memory.shape[0] > self.max_reference_size:
                self.reference_memory = self.reference_memory[
                    -self.max_reference_size:
                ]

        return drift_detected


class FeatureKS:
    """
    Unsupervised feature-wise Kolmogorov-Smirnov drift detector.

    The detector compares the current chunk with a bounded reference memory
    independently for each input feature. A drift alarm is raised when the
    fraction of features with statistically significant distributional change
    exceeds theta.
    """

    def __init__(
        self,
        input_dim,
        chunk_size,
        alpha=0.03,
        theta=0.25,
        max_reference_chunks=20,
        cooldown=15,
    ):
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.theta = theta

        self.max_reference_chunks = max_reference_chunks
        self.max_reference_size = max_reference_chunks * chunk_size

        self.cooldown = cooldown
        self.cooldown_left = 0

        self.reference_memory = None

        print(
            "FEATURE-KS CONFIG:",
            f"alpha={self.alpha}",
            f"theta={self.theta}",
            f"max_reference_chunks={self.max_reference_chunks}",
            f"cooldown={self.cooldown}",
        )

    def _append_to_reference(self, current):
        if self.reference_memory is None:
            self.reference_memory = current.copy()
        else:
            self.reference_memory = np.vstack((self.reference_memory, current))

        if self.reference_memory.shape[0] > self.max_reference_size:
            self.reference_memory = self.reference_memory[-self.max_reference_size:]

    def detect(self, X):
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return False

        current = np.asarray(X, dtype=float)

        if self.reference_memory is None:
            self.reference_memory = current.copy()
            return False

        if current.shape[0] < 2 or self.reference_memory.shape[0] < 2:
            self._append_to_reference(current)
            return False

        changed_features = 0

        for j in range(self.input_dim):
            _, p_value = ks_2samp(
                self.reference_memory[:, j],
                current[:, j],
                alternative="two-sided",
                mode="auto",
            )

            if p_value < self.alpha:
                changed_features += 1

        drift_detected = changed_features > self.theta * self.input_dim

        if drift_detected:
            self.reference_memory = current.copy()
            self.cooldown_left = self.cooldown
        else:
            self._append_to_reference(current)

        return drift_detected


class OCDD:
    def __init__(
        self,
        nu=0.05,
        size=300,
        percent=0.30,
        kernel="rbf",
        gamma="scale",
    ):
        self.nu = nu
        self.size = size
        self.percent = percent
        self.kernel = kernel
        self.gamma = gamma

        self.scaler = StandardScaler()

        self.model = None
        self.init_buffer = []
        self.window_outliers = []

        self.drift_detected = False

    def fit_model(self, X):
        X_scaled = self.scaler.fit_transform(X)

        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )

        self.model.fit(X_scaled)

    def update(self, x):
        self.drift_detected = False

        x = np.asarray(x).ravel()

        if self.model is None:
            self.init_buffer.append(x)

            if len(self.init_buffer) >= self.size:
                self.fit_model(np.asarray(self.init_buffer))
                self.init_buffer = []

            return

        x_scaled = self.scaler.transform(x.reshape(1, -1))

        pred = self.model.predict(x_scaled)[0]
        outlier_flag = 1 if pred == -1 else 0

        self.window_outliers.append(outlier_flag)

        if len(self.window_outliers) > self.size:
            self.window_outliers.pop(0)

        if len(self.window_outliers) >= self.size:
            outlier_rate = np.mean(self.window_outliers)

            if outlier_rate >= self.percent:
                self.drift_detected = True

                self.model = None
                self.window_outliers = []
                self.init_buffer = [x]


def make_classifier(name):
    if name == "gnb":
        return naive_bayes.GaussianNB()

    if name == "ht":
        return tree.HoeffdingTreeClassifier()

    raise ValueError("Unknown classifier. Use 'gnb' or 'ht'.")


def x_to_dict(x):
    return {
        f"f{j}": float(x[j])
        for j in range(len(x))
    }


def retrain_classifier_from_recent_chunks(classifier_name, recent_chunks):
    clf = make_classifier(classifier_name)

    for X_chunk, y_chunk in recent_chunks:
        for x, y_true in zip(X_chunk, y_chunk):
            clf.learn_one(x_to_dict(x), int(y_true))

    return clf


def adapt_classifier(method_data, recent_chunks):
    strategy = CONFIG["adaptation_strategy"]

    if strategy == "hard_reset":
        method_data["classifier"] = make_classifier(CONFIG["classifier"])

    elif strategy == "soft_retrain_recent":
        method_data["classifier"] = retrain_classifier_from_recent_chunks(
            CONFIG["classifier"],
            recent_chunks
        )

    elif strategy == "no_classifier_reset":
        pass

    else:
        raise ValueError(
            "Unknown adaptation_strategy. Use: "
            "'hard_reset', 'soft_retrain_recent', or 'no_classifier_reset'."
        )


def compute_segment_ids(n_chunks, alarms):
    segment_ids = np.zeros(n_chunks, dtype=int)

    current_segment = 0
    alarms_set = set(alarms)

    for i in range(n_chunks):
        segment_ids[i] = current_segment
        if i in alarms_set:
            current_segment += 1

    return segment_ids


def compute_segment_statistics(dataset_name, method_name, chunk_errors, alarms):
    errors = np.asarray(chunk_errors, dtype=float)
    n_chunks = len(errors)

    segment_ids = compute_segment_ids(n_chunks, alarms)

    global_mean = float(np.mean(errors))
    global_std = float(np.std(errors))

    rows = []

    for seg_id in sorted(np.unique(segment_ids)):
        idx = np.where(segment_ids == seg_id)[0]
        seg_errors = errors[idx]

        rows.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "SegmentID": int(seg_id),
            "StartChunk": int(idx[0]),
            "EndChunk": int(idx[-1]),
            "Length": int(len(idx)),
            "MeanError": float(np.mean(seg_errors)),
            "StdError": float(np.std(seg_errors)),
            "MinError": float(np.min(seg_errors)),
            "MaxError": float(np.max(seg_errors)),
            "GlobalMeanError": global_mean,
            "GlobalStdError": global_std,
            "StdReduction": float(global_std - np.std(seg_errors)),
            "StdRatioToGlobal": float(np.std(seg_errors) / global_std) if global_std > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def compute_stability_summary(dataset_name, method_name, chunk_errors, alarms):
    errors = np.asarray(chunk_errors, dtype=float)
    n_chunks = len(errors)

    global_std = float(np.std(errors))
    global_mean = float(np.mean(errors))

    seg_df = compute_segment_statistics(
        dataset_name=dataset_name,
        method_name=method_name,
        chunk_errors=chunk_errors,
        alarms=alarms,
    )

    valid = seg_df[seg_df["Length"] >= CONFIG["min_segment_length"]].copy()

    if len(valid) == 0:
        weighted_within_std = np.nan
        mean_segment_std = np.nan
        std_ratio = np.nan
        stable_segments_percent = np.nan
    else:
        weighted_within_std = float(
            np.average(valid["StdError"], weights=valid["Length"])
        )
        mean_segment_std = float(np.mean(valid["StdError"]))
        std_ratio = float(weighted_within_std / global_std) if global_std > 0 else np.nan
        stable_segments_percent = float(
            np.mean(valid["StdError"] < global_std) * 100.0
        )

    return {
        "Dataset": dataset_name,
        "Method": method_name,
        "GlobalMeanChunkError": global_mean,
        "GlobalStdChunkError": global_std,
        "Segments": int(seg_df["SegmentID"].nunique()),
        "ValidSegments": int(len(valid)),
        "Alarms": int(len(alarms)),
        "WeightedWithinSegmentStd": weighted_within_std,
        "MeanSegmentStd": mean_segment_std,
        "StdRatioWithinToGlobal": std_ratio,
        "StableSegmentsPercent": stable_segments_percent,
        "Interpretation": (
            "stable segmentation"
            if pd.notna(std_ratio) and std_ratio < 1.0
            else "not confirmed"
        ),
    }


def compute_before_after_alarm_analysis(dataset_name, method_name, chunk_errors, alarms):
    errors = np.asarray(chunk_errors, dtype=float)
    n_chunks = len(errors)
    w = CONFIG["before_after_window"]

    rows = []

    for alarm in alarms:
        before_start = max(0, alarm - w)
        before_end = alarm

        after_start = alarm + 1
        after_end = min(n_chunks, alarm + 1 + w)

        before = errors[before_start:before_end]
        after = errors[after_start:after_end]

        if len(before) == 0 or len(after) == 0:
            continue

        before_mean = float(np.mean(before))
        after_mean = float(np.mean(after))

        rows.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "AlarmChunk": int(alarm),
            "BeforeStart": int(before_start),
            "BeforeEnd": int(before_end - 1),
            "AfterStart": int(after_start),
            "AfterEnd": int(after_end - 1),
            "BeforeMeanError": before_mean,
            "AfterMeanError": after_mean,
            "Improvement": float(before_mean - after_mean),
            "RelativeImprovementPercent": float(
                100.0 * (before_mean - after_mean) / before_mean
            ) if before_mean > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def plot_error_with_HAST_segments(dataset_name, chunk_errors, methods, output_path):
    n_chunks = len(chunk_errors["HAST"])
    x_axis = np.arange(n_chunks)

    HAST_alarms = methods["HAST"]["alarms"]
    HAST_segments = compute_segment_ids(n_chunks, HAST_alarms)

    plt.figure(figsize=(18, 8))

    # Cieniowanie segmentów HAST.
    unique_segments = sorted(np.unique(HAST_segments))
    for seg_id in unique_segments:
        idx = np.where(HAST_segments == seg_id)[0]
        start = idx[0] - 0.5
        end = idx[-1] + 0.5
        if seg_id % 2 == 0:
            plt.axvspan(start, end, alpha=0.08)
        else:
            plt.axvspan(start, end, alpha=0.16)

    for method_name in chunk_errors.keys():
        linewidth = 2.8 if method_name == "HAST" else 1.4
        alpha = 1.0 if method_name == "HAST" else 0.75
        plt.plot(
            x_axis,
            chunk_errors[method_name],
            linewidth=linewidth,
            alpha=alpha,
            label=method_name,
        )

    for alarm in HAST_alarms:
        plt.axvline(alarm, linestyle="--", linewidth=1.5)

    plt.title(
        f"{dataset_name}: chunk error and HAST segments",
        fontsize=18,
        fontweight="bold",
    )
    plt.xlabel("Chunk index", fontsize=14, fontweight="bold")
    plt.ylabel("Chunk error", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_HAST_segment_boxplot(dataset_name, chunk_errors_HAST, alarms, output_path):
    errors = np.asarray(chunk_errors_HAST, dtype=float)
    n_chunks = len(errors)
    segment_ids = compute_segment_ids(n_chunks, alarms)

    data = []
    labels = []

    for seg_id in sorted(np.unique(segment_ids)):
        idx = np.where(segment_ids == seg_id)[0]
        if len(idx) >= CONFIG["min_segment_length"]:
            data.append(errors[idx])
            labels.append(str(seg_id))

    plt.figure(figsize=(14, 7))

    if len(data) > 0:
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.axhline(np.mean(errors), linestyle="--", linewidth=1.5)
        plt.title(
            f"{dataset_name}: HAST chunk error distribution inside segments",
            fontsize=17,
            fontweight="bold",
        )
        plt.xlabel("HAST SegmentID", fontsize=14, fontweight="bold")
        plt.ylabel("Chunk error", fontsize=14, fontweight="bold")
        plt.grid(True, axis="y", alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "Too few HAST segments for boxplot",
            ha="center",
            va="center",
            fontsize=16,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_segment_std_vs_global(dataset_name, segment_df, output_path):
    valid = segment_df[segment_df["Length"] >= CONFIG["min_segment_length"]].copy()

    plt.figure(figsize=(14, 7))

    if len(valid) > 0:
        x = np.arange(len(valid))
        labels = [str(v) for v in valid["SegmentID"].tolist()]

        plt.bar(x, valid["StdError"].values)
        plt.axhline(
            valid["GlobalStdError"].iloc[0],
            linestyle="--",
            linewidth=2.0,
            label="Global StdChunkError",
        )

        plt.xticks(x, labels)
        plt.title(
            f"{dataset_name}: HAST segment error stability",
            fontsize=17,
            fontweight="bold",
        )
        plt.xlabel("HAST SegmentID", fontsize=14, fontweight="bold")
        plt.ylabel("Std of chunk error", fontsize=14, fontweight="bold")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(fontsize=12)
    else:
        plt.text(
            0.5,
            0.5,
            "Too few HAST segments for stability analysis",
            ha="center",
            va="center",
            fontsize=16,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_before_after_alarm(dataset_name, before_after_df, output_path):
    plt.figure(figsize=(14, 7))

    if len(before_after_df) > 0:
        x = np.arange(len(before_after_df))
        labels = [str(v) for v in before_after_df["AlarmChunk"].tolist()]

        plt.plot(
            x,
            before_after_df["BeforeMeanError"].values,
            marker="o",
            linewidth=2.0,
            label="Before alarm",
        )
        plt.plot(
            x,
            before_after_df["AfterMeanError"].values,
            marker="o",
            linewidth=2.0,
            label="After alarm",
        )

        plt.xticks(x, labels)
        plt.title(
            f"{dataset_name}: HAST before/after alarm error",
            fontsize=17,
            fontweight="bold",
        )
        plt.xlabel("HAST alarm chunk", fontsize=14, fontweight="bold")
        plt.ylabel("Mean chunk error", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
    else:
        plt.text(
            0.5,
            0.5,
            "Too few HAST alarms for before/after analysis",
            ha="center",
            va="center",
            fontsize=16,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_detector_alarms(dataset_name, methods, n_chunks, output_path):
    y_positions = {
        "NO_DETECTOR": 0,
        "ADWIN": 1,
        "DDM": 2,
        "EDDM": 3,
        "OCDD": 4,
        "Feature-KS": 5,
        "HAST": 6,
    }

    plt.figure(figsize=(18, 5))

    for method_name, method_data in methods.items():
        alarms = method_data["alarms"]
        y_pos = y_positions[method_name]

        if len(alarms) > 0:
            plt.scatter(
                alarms,
                [y_pos] * len(alarms),
                s=55,
                label=method_name,
            )

    plt.xlim(-1, n_chunks)
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.xlabel("Chunk index", fontsize=14, fontweight="bold")
    plt.ylabel("Detector", fontsize=14, fontweight="bold")
    plt.title(
        f"{dataset_name}: detector alarms",
        fontsize=17,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)

    dataset_name = safe_name(CONFIG["input_file"])

    X, y = load_dataset(
        CONFIG["input_file"],
        CONFIG["label_column"],
    )

    n_samples, n_features = X.shape

    chunk_size = CONFIG["chunk_size"]
    n_chunks = n_samples // chunk_size

    if n_chunks < 2:
        raise ValueError(
            "Too few chunks. Decrease chunk_size or use a larger dataset."
        )

    print("=" * 70)
    print("REAL-WORLD HAST SEGMENT STABILITY EXPERIMENT")
    print("=" * 70)
    print(f"Input file              : {CONFIG['input_file']}")
    print(f"Samples                 : {n_samples}")
    print(f"Features                : {n_features}")
    print(f"Chunks                  : {n_chunks}")
    print(f"Chunk size              : {chunk_size}")
    print(f"Classifier              : {CONFIG['classifier']}")
    print(f"Adaptation strategy     : {CONFIG['adaptation_strategy']}")
    print(f"Adaptation memory chunks: {CONFIG['adaptation_memory_chunks']}")
    print(f"HAST cooldown           : {CONFIG['HAST_cooldown']}")
    print(f"Before/after window     : {CONFIG['before_after_window']}")
    print()

    methods = {
        "NO_DETECTOR": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": None,
            "alarms": [],
            "runtime": 0.0,
        },

        "ADWIN": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": drift.ADWIN(delta=CONFIG["adwin_delta"]),
            "alarms": [],
            "runtime": 0.0,
        },

        "DDM": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": drift.binary.DDM(),
            "alarms": [],
            "runtime": 0.0,
        },

        "EDDM": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": drift.binary.EDDM(),
            "alarms": [],
            "runtime": 0.0,
        },

        "OCDD": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": OCDD(
                nu=CONFIG["ocdd_nu"],
                size=CONFIG["ocdd_size"],
                percent=CONFIG["ocdd_percent"],
                kernel=CONFIG["ocdd_kernel"],
                gamma=CONFIG["ocdd_gamma"],
            ),
            "alarms": [],
            "runtime": 0.0,
        },

        "Feature-KS": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": FeatureKS(
                input_dim=n_features,
                chunk_size=chunk_size,
                alpha=CONFIG["feature_ks_alpha"],
                theta=CONFIG["feature_ks_theta"],
                max_reference_chunks=CONFIG["feature_ks_max_reference_chunks"],
                cooldown=CONFIG["feature_ks_cooldown"],
            ),
            "alarms": [],
            "runtime": 0.0,
        },

        "HAST": {
            "classifier": make_classifier(CONFIG["classifier"]),
            "detector": HAST(
                input_dim=n_features,
                chunk_size=chunk_size,
                alpha=CONFIG["HAST_alpha"],
                theta=CONFIG["HAST_theta"],
                r=CONFIG["HAST_r"],
                s=CONFIG["HAST_s"],
                e=CONFIG["HAST_e"],
                max_reference_chunks=CONFIG["max_reference_chunks"],
                cooldown=CONFIG["HAST_cooldown"],
            ),
            "alarms": [],
            "runtime": 0.0,
        },
    }

    chunk_errors = {m: [] for m in methods.keys()}
    predictions = {m: [] for m in methods.keys()}
    targets = {m: [] for m in methods.keys()}

    chunk_log_rows = []

    recent_chunks = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        X_chunk = X[start:end]
        y_chunk = y[start:end]

        recent_chunks.append((X_chunk.copy(), y_chunk.copy()))

        if len(recent_chunks) > CONFIG["adaptation_memory_chunks"]:
            recent_chunks.pop(0)

        # Feature-KS is chunk-based, so it is evaluated once before
        # the per-instance loop, analogously to HAST.
        feature_ks_alarm_this_chunk = False
        feature_ks_start = time.perf_counter()

        if methods["Feature-KS"]["detector"].detect(X_chunk):
            methods["Feature-KS"]["alarms"].append(chunk_idx)
            feature_ks_alarm_this_chunk = True
            adapt_classifier(methods["Feature-KS"], recent_chunks)

        methods["Feature-KS"]["runtime"] += time.perf_counter() - feature_ks_start

        # HAST jest chunkowy, więc wykrywamy go raz przed pętlą po metodach.
        HAST_alarm_this_chunk = False
        HAST_start = time.perf_counter()

        if methods["HAST"]["detector"].detect(X_chunk):
            methods["HAST"]["alarms"].append(chunk_idx)
            HAST_alarm_this_chunk = True
            adapt_classifier(methods["HAST"], recent_chunks)

        methods["HAST"]["runtime"] += time.perf_counter() - HAST_start

        for method_name, method_data in methods.items():
            clf = method_data["classifier"]
            detector = method_data["detector"]

            chunk_preds = []
            method_alarm_this_chunk = False

            method_start = time.perf_counter()

            for i in range(len(X_chunk)):
                x = X_chunk[i]
                y_true = int(y_chunk[i])

                x_dict = x_to_dict(x)

                y_pred = clf.predict_one(x_dict)

                if y_pred is None:
                    y_pred = y_true

                error = 0 if y_pred == y_true else 1

                chunk_preds.append(y_pred)

                predictions[method_name].append(y_pred)
                targets[method_name].append(y_true)

                clf.learn_one(x_dict, y_true)

                if method_name in ["ADWIN", "DDM", "EDDM"]:
                    detector.update(error)

                    if detector.drift_detected:
                        if len(method_data["alarms"]) == 0 or method_data["alarms"][-1] != chunk_idx:
                            method_data["alarms"].append(chunk_idx)
                            method_alarm_this_chunk = True
                            adapt_classifier(method_data, recent_chunks)

                if method_name == "OCDD":
                    detector.update(x)

                    if detector.drift_detected:
                        if len(method_data["alarms"]) == 0 or method_data["alarms"][-1] != chunk_idx:
                            method_data["alarms"].append(chunk_idx)
                            method_alarm_this_chunk = True
                            adapt_classifier(method_data, recent_chunks)

            method_data["runtime"] += time.perf_counter() - method_start

            chunk_error = float(np.mean(np.asarray(chunk_preds) != y_chunk))
            chunk_errors[method_name].append(chunk_error)

            if method_name == "HAST":
                alarm_flag = int(HAST_alarm_this_chunk)
            elif method_name == "Feature-KS":
                alarm_flag = int(feature_ks_alarm_this_chunk)
            elif method_name in ["ADWIN", "DDM", "EDDM", "OCDD"]:
                alarm_flag = int(method_alarm_this_chunk)
            else:
                alarm_flag = 0

            chunk_log_rows.append({
                "Dataset": dataset_name,
                "Method": method_name,
                "Chunk": int(chunk_idx),
                "ChunkStartSample": int(start),
                "ChunkEndSample": int(end - 1),
                "ChunkError": chunk_error,
                "Alarm": alarm_flag,
            })

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    results = []

    for method_name in methods.keys():
        y_true = np.asarray(targets[method_name])
        y_pred = np.asarray(predictions[method_name])

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)

        prequential_error = 1.0 - acc
        mean_chunk_error = np.mean(chunk_errors[method_name])
        std_chunk_error = np.std(chunk_errors[method_name])

        alarms = methods[method_name]["alarms"]

        alarms_per_1000 = (
            len(alarms) / n_samples
        ) * 1000.0

        if len(alarms) >= 2:
            mean_alarm_gap = float(np.mean(np.diff(alarms)))
        else:
            mean_alarm_gap = np.nan

        results.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "Accuracy": acc,
            "BalancedAcc": bal_acc,
            "F1_macro": f1,
            "Kappa": kappa,
            "PreqError": prequential_error,
            "MeanChunkError": mean_chunk_error,
            "StdChunkError": std_chunk_error,
            "Alarms": len(alarms),
            "Alarms/1000": alarms_per_1000,
            "MeanAlarmGap": mean_alarm_gap,
            "RuntimeSeconds": methods[method_name]["runtime"],
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    # ------------------------------------------------------------
    # CHUNK LOG + SEGMENT ID
    # ------------------------------------------------------------
    chunk_log_df = pd.DataFrame(chunk_log_rows)

    HAST_alarms = methods["HAST"]["alarms"]
    HAST_segment_ids = compute_segment_ids(n_chunks, HAST_alarms)

    chunk_log_df["HAST_SegmentID"] = chunk_log_df["Chunk"].apply(
        lambda c: int(HAST_segment_ids[int(c)])
    )

    # ------------------------------------------------------------
    # HAST SEGMENT STABILITY
    # ------------------------------------------------------------
    HAST_segment_df = compute_segment_statistics(
        dataset_name=dataset_name,
        method_name="HAST",
        chunk_errors=chunk_errors["HAST"],
        alarms=HAST_alarms,
    )

    stability_summary = compute_stability_summary(
        dataset_name=dataset_name,
        method_name="HAST",
        chunk_errors=chunk_errors["HAST"],
        alarms=HAST_alarms,
    )

    stability_summary_df = pd.DataFrame([stability_summary])

    before_after_df = compute_before_after_alarm_analysis(
        dataset_name=dataset_name,
        method_name="HAST",
        chunk_errors=chunk_errors["HAST"],
        alarms=HAST_alarms,
    )

    print()
    print("=" * 70)
    print("HAST SEGMENT STABILITY SUMMARY")
    print("=" * 70)
    print(stability_summary_df)

    print()
    print("=" * 70)
    print("HAST SEGMENTS")
    print("=" * 70)
    print(HAST_segment_df)

    if len(before_after_df) > 0:
        print()
        print("=" * 70)
        print("HAST BEFORE/AFTER ALARM ANALYSIS")
        print("=" * 70)
        print(before_after_df)

    # ------------------------------------------------------------
    # # CSV OUTPUT
    # ------------------------------------------------------------
    results_csv = os.path.join(
        CONFIG["results_dir"],
        f"{dataset_name}_global_results.csv",
    )

    chunk_log_csv = os.path.join(
        CONFIG["results_dir"],
        f"{dataset_name}_chunk_log.csv",
    )

    HAST_segments_csv = os.path.join(
        CONFIG["results_dir"],
        f"{dataset_name}_HAST_segments.csv",
    )

    stability_csv = os.path.join(
        CONFIG["results_dir"],
        f"{dataset_name}_HAST_stability_summary.csv",
    )

    before_after_csv = os.path.join(
        CONFIG["results_dir"],
        f"{dataset_name}_HAST_before_after_alarm.csv",
    )

    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    chunk_log_df.to_csv(chunk_log_csv, index=False, encoding="utf-8-sig")
    HAST_segment_df.to_csv(HAST_segments_csv, index=False, encoding="utf-8-sig")
    stability_summary_df.to_csv(stability_csv, index=False, encoding="utf-8-sig")
    before_after_df.to_csv(before_after_csv, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # # PLOTS
    # ------------------------------------------------------------
    plot_error_path = os.path.join(
        CONFIG["plots_dir"],
        f"{dataset_name}_01_error_with_HAST_segments.png",
    )

    plot_box_path = os.path.join(
        CONFIG["plots_dir"],
        f"{dataset_name}_02_HAST_segment_boxplot.png",
    )

    plot_std_path = os.path.join(
        CONFIG["plots_dir"],
        f"{dataset_name}_03_HAST_segment_std_vs_global.png",
    )

    plot_before_after_path = os.path.join(
        CONFIG["plots_dir"],
        f"{dataset_name}_04_HAST_before_after_alarm.png",
    )

    plot_alarms_path = os.path.join(
        CONFIG["plots_dir"],
        f"{dataset_name}_05_detector_alarms.png",
    )

    plot_error_with_HAST_segments(
        dataset_name=dataset_name,
        chunk_errors=chunk_errors,
        methods=methods,
        output_path=plot_error_path,
    )

    plot_HAST_segment_boxplot(
        dataset_name=dataset_name,
        chunk_errors_HAST=chunk_errors["HAST"],
        alarms=HAST_alarms,
        output_path=plot_box_path,
    )

    plot_segment_std_vs_global(
        dataset_name=dataset_name,
        segment_df=HAST_segment_df,
        output_path=plot_std_path,
    )

    plot_before_after_alarm(
        dataset_name=dataset_name,
        before_after_df=before_after_df,
        output_path=plot_before_after_path,
    )

    plot_detector_alarms(
        dataset_name=dataset_name,
        methods=methods,
        n_chunks=n_chunks,
        output_path=plot_alarms_path,
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Global results          : {results_csv}")
    print(f"Chunk log               : {chunk_log_csv}")
    print(f"HAST segments           : {HAST_segments_csv}")
    print(f"HAST stability summary  : {stability_csv}")
    print(f"HAST before/after alarms: {before_after_csv}")
    print(f"Plot 1                  : {plot_error_path}")
    print(f"Plot 2                  : {plot_box_path}")
    print(f"Plot 3                  : {plot_std_path}")
    print(f"Plot 4                  : {plot_before_after_path}")
    print(f"Plot 5                  : {plot_alarms_path}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    ratio = stability_summary["StdRatioWithinToGlobal"]
    if pd.notna(ratio) and ratio < 1.0:
        print(
            "HAST segmentation is supported: "
            "weighted within-segment error variability is lower than global variability."
        )
    else:
        print(
            "HAST segmentation is not confirmed by this criterion: "
            "within-segment variability is not lower than global variability."
        )

    if len(before_after_df) > 0:
        mean_improvement = before_after_df["Improvement"].mean()
        print(f"Mean before-after alarm improvement: {mean_improvement:.6f}")

        if mean_improvement > 0:
            print(
                "After HAST alarms, the mean chunk error tends to decrease. "
                "This supports useful segmentation/adaptation."
            )
        else:
            print(
                "After HAST alarms, the mean chunk error does not decrease. "
                "This weakens the segmentation/adaptation argument."
            )
    else:
        print(
            "Before-after alarm analysis was not possible because there were too few HAST alarms "
            "or alarms were too close to stream boundaries."
        )


if __name__ == "__main__":
    main()
