import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

# --- Funktionen definieren ---
def f1(x, y, lam=0):
    if x - y > 0:
        return (0.66 + lam) * (x - y)
    else:
        return (0.33 + lam) * (x - y)

def f2(x, y, lam=0):
    return 0.66 * x + -0.33 * y + lam * (x - y)

# --- Daten vorbereiten ---
for lam in [0, -0.5, 0.5]:

    data_x = np.round(np.linspace(0, 100, 201), 2)
    data_y = np.round(np.linspace(0, 100, 201), 2)
    data = {"x": [], "y": [], "f1": [], "f2": [], "diff": []}

    for x, y in product(data_x, data_y):
        val_f1 = f1(x, y, lam)
        val_f2 = f2(x, y, lam)
        data["x"].append(x)
        data["y"].append(y)
        data["f1"].append(val_f1)
        data["f2"].append(val_f2)
        data["diff"].append(val_f1 - val_f2)

    df = pd.DataFrame(data)
    cmap = "coolwarm"
    vmin, vmax = -15, 15
    xlim = (0, 30)
    ylim = (0, 30)

    # --- Pivot-Tabellen für Heatmaps ---
    dff1 = df.pivot(index="x", columns="y", values="f1")
    dff2 = df.pivot(index="x", columns="y", values="f2")
    dfdiff = df.pivot(index="x", columns="y", values="diff")

    # --- Heatmaps plotten ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(dff1, ax=axs[0], cbar=True, center=0, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Real costs")
    axs[0].set(xlim=xlim, ylim=ylim)

    sns.heatmap(dff2, ax=axs[1], cbar=True, center=0, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title("Surrogate")
    axs[1].set(xlim=xlim, ylim=ylim)

    sns.heatmap(dfdiff, ax=axs[2], cbar=True, center=0, cmap=cmap, vmin=vmin,
                vmax=vmax)
    axs[2].set_title("Underestimation (real - surrogate)")
    axs[2].set(xlim=xlim, ylim=ylim)

    plt.tight_layout()
    plt.show()

    result_dir = Path(".") / "results"
    plt.savefig(result_dir / f"objective_comparison_lam_{lam}.png")