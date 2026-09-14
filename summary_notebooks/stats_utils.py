import os
import zlib
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

OUT      = Path("outputs")
TAB_OUT  = OUT / "tables"
BASE_OUT = Path("outputs")

SUFFIX_MAP = {1: "frozen", 0: "fine_tune"}


IN_DOMAIN_MODELS = ["nako_mae", "nako_mae_scr","simclr_inet_nako", "simclr_nako",
                     "retfound", "dino_nako", ]
OUT_OF_DOMAIN    = "dino_meta"
SUPERVISED_BS    = ["resnet_imagenet", "scratch"]
MAE_MODELS       = ["nako_mae", "retfound", "nako_mae_scr"]
SIMCLR_MODELS    = ["simclr_nako", "simclr_inet_nako"]
FM_MODELS        = ["retfound", "dino_meta", "dino_nako"]
SSL_MODELS       = ["simclr_nako", "simclr_inet_nako", "nako_mae", "nako_mae_scr"]
ALL_MODELS       = IN_DOMAIN_MODELS + [OUT_OF_DOMAIN] + SUPERVISED_BS
ALPHA            = 0.05
MODEL_LABELS = {
    "nako_mae":         "MAE-NAKO-ImageNet",
    "nako_mae_scr":     "MAE-NAKO",
    "simclr_inet_nako": "SimCLR-INet-NAKO",
    "simclr_nako":      "SimCLR-NAKO",
    "retfound":         "RETFound",
    "dino_nako":        "DINOv2-NAKO",
    "dino_meta":        "DINOv2-Meta",
    "resnet_imagenet":  "ResNet-ImageNet",
    "scratch":          "ResNet-Scratch",
}
FAMILY_MAP = {
    "simclr_nako":      "SSL",
    "simclr_inet_nako": "SSL",
    "dino_nako":        "FM",
    "dino_meta":        "FM",
    "retfound":         "FM",
    "nako_mae":         "SSL",
    "nako_mae_scr": "SSL",
    "resnet_imagenet":  "Supervised",
    "scratch":          "Supervised",
}

def _stable_seed(*parts: str) -> int:
    
    key = "_".join(str(p) for p in parts)
    return zlib.crc32(key.encode()) % (2**31)

def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:

    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(values)), float(np.percentile(boot_means, 2.5)), \
           float(np.percentile(boot_means, 97.5))


def table4_min_training_size(df: pd.DataFrame, freeze_encoder = None,
                              abs_threshold_ci: float = 0.75,
                              peak_fraction: float = 0.95,
                              n_boot: int = 2000, surv_head = "mlp", 
                              table_number = 4) -> pd.DataFrame:
    protocol_label = None
    if freeze_encoder == 1:
        protocol_label = 'freeze'
    elif freeze_encoder == 0:
        protocol_label = 'fine-tune'
    else:
        raise ValueError('unknown')

    train_sizes = sorted(df.train_size.unique())
    max_ts      = max(train_sizes)
    rows        = []

    for model in ALL_MODELS:
        sub = df[(df.model == model) & (df.survival_head == surv_head) & (df.freeze_encoder == freeze_encoder)]
        if sub.empty:
            continue

        boot_stats = {}
        for ts in train_sizes:
            vals = sub[sub.train_size == ts]["ci_index"].to_numpy()
            mean, lo, hi = _bootstrap_mean_ci(
                vals, n_boot=n_boot, seed=_stable_seed(model, ts)
            )
            boot_stats[ts] = (mean, lo, hi)

        lo_series = pd.Series({ts: boot_stats[ts][1] for ts in train_sizes})

        means_by_ts = {ts: boot_stats[ts][0] for ts in train_sizes
                       if not np.isnan(boot_stats[ts][0])}
        if means_by_ts:
            peak_ts = max(means_by_ts, key=means_by_ts.get)
            peak_mean, peak_lo, peak_hi = boot_stats[peak_ts]
        else:
            peak_ts = None
            peak_mean, peak_lo, peak_hi = np.nan, np.nan, np.nan

        abs_met   = lo_series[lo_series >= abs_threshold_ci]
        min_n_abs = f"\u2264{abs_met.index.min():,}" if not abs_met.empty else f">{max_ts:,}"

        if not np.isnan(peak_mean):
            rel_threshold = peak_fraction * peak_mean
            rel_met       = lo_series[lo_series >= rel_threshold]
            min_n_rel     = f"\u2264{rel_met.index.min():,}" if not rel_met.empty else f">{max_ts:,}"
        else:
            min_n_rel = "\u2014"


        peak_str = (
            f"{peak_mean:.3f} [{peak_lo:.3f}, {peak_hi:.3f}]"
            if not np.isnan(peak_mean) else "\u2014"
        )

        rows.append({
            "Model":                                                       MODEL_LABELS.get(model, model),
            "Family":                                                      FAMILY_MAP.get(model, ""),
            f"Min n (CI lower\u2265{abs_threshold_ci})":                    min_n_abs,
            f"Min n (CI lower\u2265{int(peak_fraction*100)}% peak)":        min_n_rel,
            "Peak C-index [95% CI]":                           peak_str,
        })

    result = pd.DataFrame(rows)
    fname  = f"table4_min_training_size_{surv_head}_{protocol_label.lower().replace(' ','-')}"
   
    return result


def prepare_df(raw_df: pd.DataFrame, freeze_encoder: int = 1) -> pd.DataFrame:
   
    df = raw_df.copy()
    df = df.rename(columns={
        "test/concordance_index": "ci_index",
        "test/ibs":               "ibs",
        "summary":                "model",
    })
    df["train_size"] = df["train_size"].astype(int)

    if "freeze_encoder" in df.columns:
        df = df[df["freeze_encoder"] == freeze_encoder].copy()
        protocol_label = "frozen" if freeze_encoder == 1 else "fine-tuned"
        print(f"Protocol: {protocol_label} (freeze_encoder={freeze_encoder})")
    else:
        print("'freeze_encoder' column not found — no protocol filter applied")

    df = df[["survival_head", "ci_index", "ibs", "train_size", "model", "seed", "freeze_encoder"]]

    next_seed = int(df.seed.max() + 1)
    df["seed"] = df["seed"].fillna(next_seed).astype(int)

    known_models = IN_DOMAIN_MODELS + [OUT_OF_DOMAIN] + SUPERVISED_BS
    df = df[df.model.isin(known_models)].reset_index(drop=True)

    df["seed"] = df.groupby(["model", "train_size", "survival_head"])["seed"] \
                   .transform(lambda x: pd.factorize(x)[0])

    print(f"Rows after filtering : {len(df)}")
    print(f"Models               : {sorted(df.model.unique())}")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"WARNING — nulls found:\n{null_counts[null_counts > 0]}")
    else:
        print("No nulls found.")

    seed_counts = df.groupby(["model", "train_size", "survival_head"])["seed"].nunique()
    if (seed_counts < 5).any():
        print("WARNING — some conditions have fewer than 5 seeds:")
        print(seed_counts[seed_counts < 5].to_string())
    else:
        print("All conditions have 5 seeds.")

    return df



def compute_spread(data, objective_set_name, objectives=None):
    d = data if objectives is None else data[data["objective"].isin(objectives)]

    out = (
        d.groupby(["protocol", "survival_head", "train_size"])
         .agg(
             cindex_spread=("cindex_fit", lambda x: x.max() - x.min()),
             ibs_spread=("ibs_fit", lambda x: x.max() - x.min())
         )
         .reset_index()
    )

    out["objective_set"] = objective_set_name
    return out

def get_axis_limits_df(df, metric, margin=0.01):
    """Same logic as get_axis_limits, adapted for a flat dataframe."""
    all_y = df[metric].values
    y_min = np.floor((all_y.min() - 0.001) * 1000) / 1000
    y_max = np.ceil((all_y.max() + margin) * 1000) / 1000
    return y_min, y_max

def style_xaxis(ax):
    ax.set_xscale("log")  
    ax.set_xlim(80, 1.2e5)

    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e5]))
    ax.xaxis.set_major_formatter(FixedFormatter([r'$10^2$', r'$10^5$']))
    ax.xaxis.set_minor_locator(NullLocator())   

    ax.tick_params(axis='both', labelsize=10, width=1, length=2)


def plot_summary(    
    metric_fit_val = 'cindex_fit',   
    spread_val = 'cindex_spread',
    fontsize = 10,
    text_x = -0.12, 
    text_y = 1.05,   
    y_legend = -0.3, 
    legend_below = True,   
    top_yticks = None,
    bottom_yticks = None,
    save_path = None,
    layout = "1x4",     
    pred = None,
    spread = None   
):  

    protocols = ["frozen", "fine-tuned"]
    heads = ["linear", "mlp"]

    y_label_names = {'cindex_fit':'Predicted C-index',
                     'ibs_fit': 'Predicted IBS'}

    
    group_colors = {
        'supervised BS':  '#1565C0',
        'out-of-domain FM': '#05771D',
        'in-domain PR':   '#6A1B9A',
        'in-domain FM':   '#050606',
    }
    group_shades = {
        'supervised BS':  ['#1565C0', '#42A5F5'],
        'out-of-domain FM': ['#05771D'],
        'in-domain PR':   ['#6A1B9A', '#D81B60', '#F3D0DC'],
        'in-domain FM':   ['#000000', '#90A4AE'],
    }

    objective_colors = {
        "SimCLR": group_shades['in-domain PR'][1],   
        "MAE": group_shades['in-domain PR'][0],       
        "DINOv2": group_colors['out-of-domain FM'],   
    }
    head_styles = {
        "mlp": "-",
        "linear": "--",
    }

     
    objective_set_colors = {
    "all objectives": "tab:blue",
    "simclr / mae / dino only": "tab:orange",
}

    head_labels = {
    "linear": "Linear",
    "mlp": "MLP",
}

    stylef = 'mpl.mplstyle'
    width_pt = 451
    width_in = width_pt / 72.27

    with plt.style.context(stylef):

        if layout == "1x4":
            
            fig, axes = plt.subplots(
                1, 4,
                figsize=(width_in, 2.6),   
                constrained_layout=True,
            )
            bottom_axes    = {"frozen": axes[0], "fine-tuned": axes[1]}
            top_axes = {"frozen": axes[2], "fine-tuned": axes[3]}
            legend_ref_ax = axes[3]     
            panel_labels = ['A', 'B', 'C', 'D']
            flat_axes = list(axes)
        else:  
            fig, axes = plt.subplots(
                2, 2,
                figsize=(width_in, 5),
                constrained_layout=True,
            )
            top_axes    = {"frozen": axes[0, 0], "fine-tuned": axes[0, 1]}
            bottom_axes = {"frozen": axes[1, 0], "fine-tuned": axes[1, 1]}
            legend_ref_ax = axes[1, 1]
            panel_labels = ['A', 'B', 'C', 'D']
            flat_axes = list(axes.flat)

        def style_fig_legend(**kwargs):
            leg = fig.legend(frameon=False, fontsize=fontsize, **kwargs)
            for text in leg.get_texts():
                text.set_fontname("Arial")
            leg.get_title().set_fontname("Arial")
            leg.get_title().set_fontsize(fontsize)
            leg.get_title().set_fontweight("bold")
            return leg

        for ax in flat_axes:
            style_xaxis(ax)

        if top_yticks is None:
            top_y_min, top_y_max = get_axis_limits_df(pred, metric_fit_val)
            top_yticks = np.linspace(top_y_min, top_y_max, 5)
        if bottom_yticks is None:
            bottom_y_min, bottom_y_max = get_axis_limits_df(spread, spread_val)
            bottom_yticks = np.linspace(bottom_y_min, bottom_y_max, 5)

 
        for i, protocol in enumerate(protocols):
            ax = top_axes[protocol]

            for head in heads:
                subset = pred[
                    (pred.protocol == protocol)
                    & (pred.survival_head == head)
                ]
                for objective in ["SimCLR", "MAE", "DINOv2"]:
                    d = subset[subset.objective == objective].sort_values("train_size")
                    ax.plot(
                        d.train_size, d[metric_fit_val],
                        color=objective_colors[objective],
                        linestyle=head_styles[head],
                        linewidth=2,
                    )

            sns.despine(ax=ax, offset={'left': 3, 'bottom': 3}, trim=False)
            ax.set_yticks(top_yticks)
            ax.set_ylim(top_yticks[0], top_yticks[-1])
            if i == 0:
                ax.set_ylabel(y_label_names[metric_fit_val], fontsize=fontsize, fontname='Arial')
            else:
                ax.set_ylabel("")
        for j, protocol in enumerate(protocols):


            ax = bottom_axes[protocol]

            sub = spread[spread.protocol == protocol]
            for head in heads:
                for objective_set in ["all objectives", "simclr / mae / dino only"]:
                    d = sub[
                        (sub.survival_head == head)
                        & (sub.objective_set == objective_set)
                    ].sort_values("train_size")
                    ax.plot(
                        d.train_size, d[spread_val],
                        color=objective_set_colors[objective_set],
                        linestyle=head_styles[head],
                        linewidth=2,
                    )

            sns.despine(ax=ax, offset={'left': 3, 'bottom': 3}, trim=False)
            ax.set_yticks(bottom_yticks)
            ax.set_ylim(bottom_yticks[0], bottom_yticks[-1])
            if j == 0:
                ax.set_ylabel("Objective spread", fontsize=fontsize, fontname='Arial')
            else:
                ax.set_ylabel("")

        for a, label in zip(flat_axes, panel_labels):
            a.text(
                text_x, text_y, f'{label}',
                transform=a.transAxes,
                fontsize=fontsize, fontweight='bold',
                ha='left', va='top', clip_on=False, fontname='Arial',
            )

        for ax in flat_axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname("Arial")

        head_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle=head_styles[h], label=head_labels[h])
            for h in heads
        ]
        objective_handles = [
            Line2D([0], [0], color=objective_colors[o], lw=2, label=o)
            for o in ["SimCLR", "MAE", "DINOv2"]
        ]
        set_handles = [
            Line2D([0], [0], color=objective_set_colors[s], lw=2,
                   label=("All objectives" if s == "all objectives" else "SimCLR / MAE / DINOv2"))
            for s in objective_set_colors
        ]

        if legend_below:
            if layout == '1x4':
                n_col =1
            else:
                n_col = 1
            style_fig_legend(handles=head_handles, title="Head", loc="upper center",
                              bbox_to_anchor=(0.15, y_legend), ncol=n_col)
            style_fig_legend(handles=objective_handles, title="Objective", loc="upper center",
                              bbox_to_anchor=(0.8, y_legend), ncol=n_col)
            leg_set = style_fig_legend(handles=set_handles, title="Objective set", loc="upper center",
                                        bbox_to_anchor=(0.48, y_legend), ncol=n_col)
            leg_set.get_title().set_ha("left")
            leg_set.get_title().set_position((-35, -0.3))
        else:
            leg1 = legend_ref_ax.legend(handles=objective_handles, title="Objective", loc="upper right")
            leg2 = legend_ref_ax.legend(handles=head_handles, title="Head", loc="lower right")
            legend_ref_ax.add_artist(leg1)

    fig.supxlabel('Longitudinal training set size (#images)', fontsize=fontsize,
                  x=0.5, y=-0.065, fontname='Arial')

    if save_path is not None:
        fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')

    plt.show()