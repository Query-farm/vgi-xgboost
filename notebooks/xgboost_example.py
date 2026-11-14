"""Parameter guards and presentation of SQL-returned XGBoost results."""

import matplotlib.pyplot as plt
from vgi_notebooks import finish

SOURCE = "Source: explicitly synthetic inputs · VGI XGBoost v0.1.0 + visible SQL\nAuthored data only; no observed outcomes or causal effect estimates"


def validate_depth(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("MAX_DEPTH must be an integer from 1 through 8")


def validate_feature(value):
    if value not in {"distance_km", "stops"}:
        raise ValueError("SWEEP_FEATURE must be distance_km or stops")


def validate_folds(value):
    if type(value) is not int or value not in {3, 6}:
        raise ValueError("CV_FOLDS must be integer 3 or 6")


def prediction_errors(points, metrics, MAX_DEPTH):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    lo = min(points.minutes.min(), points.prediction.min()) - 5
    hi = max(points.minutes.max(), points.prediction.max()) + 5
    for ax, split, color in zip(
        axes, ["Training", "Held out"], ["#267358", "#c57b40"], strict=True
    ):
        group = points[points.split == split]
        stat = metrics[metrics.split == split].iloc[0]
        ax.scatter(
            group.minutes,
            group.prediction,
            s=35,
            alpha=0.7,
            color=color,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#687b70", linewidth=1.2)
        ax.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel="Synthetic target · Minutes",
            title=f"{split} · {len(group)} routes",
        )
        ax.text(
            0.04,
            0.94,
            f"RMSE {stat.rmse_minutes:.2f} min\nMean baseline {stat.baseline_rmse_minutes:.2f} min",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Model prediction · Minutes")
    finish(
        fig,
        "Training fit is only half of the evaluation",
        f"Maximum depth {MAX_DEPTH} · 80 rounds · Identical axes; dashed line is exact prediction",
        SOURCE,
        name="01-prediction-errors",
    )


def prediction_contributions(points, contributions, MAX_DEPTH):
    row = points[points.route_id == 200].iloc[0]
    base = float(contributions.base_value.iloc[0])
    current = base
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = (
        ["Base"]
        + [
            {"distance_km": "Distance", "stops": "Stops", "rush": "Rush flag"}[f]
            for f in contributions.feature
        ]
        + ["Prediction"]
    )
    ax.bar(0, base, color="#728a7a", width=0.6)
    ax.text(
        0,
        base + 2,
        f"{base:.2f}",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.95},
    )
    for i, value in enumerate(contributions.shap_value, start=1):
        next_value = current + value
        ax.plot([i - 1 + 0.3, i - 0.3], [current, current], color="#9aaba0", linewidth=1)
        ax.bar(
            i,
            abs(value),
            bottom=min(current, next_value),
            color="#c57b40" if value >= 0 else "#456fba",
            width=0.6,
        )
        ax.text(
            i,
            max(current, next_value) + 2,
            f"{value:+.2f}",
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.95},
        )
        current = next_value
    ax.plot([3.3, 3.7], [current, current], color="#9aaba0", linewidth=1)
    ax.bar(4, row.prediction, color="#267358", width=0.6)
    ax.text(
        4,
        row.prediction + 2,
        f"{row.prediction:.2f}",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.95},
    )
    ax.axhline(
        row.minutes,
        color="#956f94",
        linestyle="--",
        linewidth=1.5,
        label=f"Synthetic target: {row.minutes:.2f} min",
    )
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("Prediction and contributions · Minutes")
    ax.set_ylim(0, max(base, current, row.minutes) + 15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    finish(
        fig,
        "Contributions explain the prediction, including its error",
        f"Held-out route 200 · {row.distance_km:.1f} km, {int(row.stops)} stops, rush {int(row.rush)} · Depth {MAX_DEPTH}",
        SOURCE,
        name="02-prediction-contributions",
    )


def feature_questions(data):
    names = {
        "distance_km": "Distance",
        "stops": "Stops",
        "rush": "Rush flag",
        "constant_control": "Constant control",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    y = list(range(len(data)))
    axes[0].barh(y, data.gain_share_percent, color="#397d62", height=0.56)
    axes[1].barh(
        y,
        data.rmse_increase_minutes,
        xerr=data.shuffle_sd_minutes,
        color="#b46b45",
        height=0.56,
        capsize=4,
    )
    axes[0].set_yticks(y, [names[x] for x in data.feature])
    axes[0].invert_yaxis()
    for ax, values, decimals in [
        (axes[0], data.gain_share_percent, 1),
        (axes[1], data.rmse_increase_minutes, 2),
    ]:
        ax.axvline(0, color="#627a70", linewidth=1)
        for i, value in enumerate(values):
            ax.annotate(
                f"{value:.{decimals}f}",
                (value + (float(data.shuffle_sd_minutes.iloc[i]) if ax is axes[1] else 0), i),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )
        ax.margins(x=0.22)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Share of total reported gain · %")
    axes[1].set_xlabel("RMSE increase after shuffling · minutes")
    axes[0].set_title("Training tree structure")
    axes[1].set_title("48 held-out rows · mean ± shuffle SD")
    finish(
        fig,
        "A feature can matter in different ways",
        "Depth 2 · ten shuffles · zeros retained · panel units differ",
        SOURCE,
        name="03-feature-questions",
    )


def partial_curve(data, feature):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(data.feature_value, data.partial_dependence, "o-", color="#397d62", markersize=5)
    ax.set_xlabel(
        "Replacement distance · km" if feature == "distance_km" else "Replacement stops · count"
    )
    ax.set_ylabel("Average model prediction · minutes")
    if feature == "stops":
        ax.set_xticks(data.feature_value)
    finish(
        fig,
        "Replace one input; average over the same background",
        f"{len(data)} returned grid values · 192 training background rows · depth 2",
        SOURCE,
        name="04-partial-dependence",
    )


def selection_scores(folds, grid, holdout, cv):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.15, 1.2]})
    axes[0].bar(folds.fold, folds.rmse, color="#a5bfb6")
    axes[0].set_xticks(folds.fold)
    axes[0].set_xlabel("Fold number")
    axes[0].set_title("Fixed depth 2 · OOF")
    axes[0].set_ylabel("RMSE · synthetic target units")
    axes[1].bar(
        range(len(grid)),
        grid.mean_cv_rmse,
        yerr=grid.fold_sd,
        capsize=4,
        color=["#397d62" if x else "#a5bfb6" for x in grid.selected],
    )
    axes[1].set_xticks(range(len(grid)), [f"Depth {d}" for d in grid.depth])
    axes[1].set_xlabel("Grid candidates")
    axes[1].set_title("Selection · mean ± fold SD")
    for i, row in enumerate(grid.itertuples()):
        axes[1].text(
            i,
            row.mean_cv_rmse + row.fold_sd + 1,
            "Selected" if row.selected else f"Rank {row.rank}",
            ha="center",
            fontsize=10,
        )
    values = [float(holdout.selected_rmse.iloc[0]), float(holdout.mean_baseline_rmse.iloc[0])]
    axes[2].bar([0, 1], values, color=["#397d62", "#b46b45"])
    axes[2].set_xticks([0, 1], ["Selected\nmodel", "Development\nmean"])
    axes[2].set_title("Reserved · 48 rows")
    for i, value in enumerate(values):
        axes[2].text(i, value + 1, f"{value:.2f}", ha="center")
    upper = max(max(folds.rmse), max(grid.mean_cv_rmse + grid.fold_sd)) * 1.35
    for ax in axes[:2]:
        ax.set_ylim(0, upper)
    axes[2].set_ylim(0, max(values) * 1.2)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    finish(
        fig,
        "Keep candidate selection and reserved evaluation distinct",
        f"{cv} equal folds on 192 development rows · same RMSE units; reserved panel uses wider y-scale · SD is not a confidence interval",
        SOURCE,
        name="05-selection-scores",
    )
