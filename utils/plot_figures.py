import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def make_publication_legend(ax, cm_colors, model_groups, group_colors, outside=True):
    """
    Two-level legend: group label (colored patch) + indented model entries.
    outside=True places it to the right of the axes.
    """
    handles = []
    labels  = []

    group_members = {}
    for model, group in sorted(model_groups.items()):
        group_members.setdefault(group, [])
        group_members[group].append(model)

    group_order = ['in-domain FM', 'out-of-domain FM', 'in-domain PR', 'supervised BS']

    for group in group_order:
        members = sorted(group_members.get(group, []))

        # group header: filled patch
        handles.append(mpatches.Patch(color=group_colors[group], alpha=0.25))
        labels.append(f'{group}')

        # model entries: solid line
        for model in members:
            color = cm_colors.get(model + '_lin', '#000000')
            handles.append(Line2D([0], [0], color=color, linestyle='-',
                                  linewidth=1.2))
            labels.append(f'  {model}')

        # blank spacer between groups
        handles.append(Line2D([0], [0], color='none'))
        labels.append('')

    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=6,
        handlelength=2.0,
        handleheight=1.0,
        handletextpad=0.5,
        borderpad=0.8,
        labelspacing=0.3,
    )

    if outside:
        leg = ax.legend(**legend_kwargs,
                        bbox_to_anchor=(1.02, 0.8),
                        loc='upper left',
                        borderaxespad=0.)
    else:
        leg = ax.legend(**legend_kwargs, loc='best')

    new_names = {
        'dino_nako':       'DINOv2 NAKO',
        'dino_meta':       'DINOv2 LVD',
        'retfound':        'RETFound',
        'resnet_INet':     'ResNet INet',
        'mae_nako':        'MAE NAKO',
        'simclr_nako':     'SimCLR NAKO',
        'simclr_INet_nako':'SimCLR INet NAKO',
        'resnet_scratch':  'ResNet Scratch',
    }

    for text, label in zip(leg.get_texts(), labels):
        if not label.startswith('  ') and label != '':
            text.set_fontweight('bold')
        new_label = label
        for old, new in new_names.items():
            if old in new_label:
                new_label = new_label.replace(old, new)
        text.set_text(new_label)

    return leg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_base_key(plot_key):
    """'retfound_mlp_fet' -> 'retfound_mlp'"""
    return plot_key.rsplit('_', 1)[0]


def get_axis_limits(all_plots, metrics, x_col='train_size', margin=0.01):
    """Compute y_min, y_max, and xticks from the data."""
    all_y = []
    all_x = set()

    for value_ in all_plots.values():
        agg = value_.groupby(x_col)[metrics].mean()
        all_y.extend(agg.values)
        all_x.update(agg.index.tolist())

    print(f"min y-axis {min(all_y):.4f}, max y-axis {max(all_y):.4f}")
    y_min = np.floor((min(all_y) - 0.001) * 1000) / 1000
    y_max = np.ceil ((max(all_y) + margin)  * 1000) / 1000
    xticks = sorted(all_x)

    return y_min, y_max, xticks


def round_down(value, decimals):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def get_yticks(y_min, y_max, n_ticks=6, define_ticks=True,
               use_auto=False, ticks=None):
    if ticks is not None:
        return ticks

    if use_auto:
        raw_step  = (y_max - y_min) / n_ticks
        magnitude = 10 ** np.floor(np.log10(raw_step))
        clean_step = np.round(raw_step / magnitude) * magnitude
        ticks_ = np.arange(
            np.ceil(y_min / clean_step) * clean_step,
            y_max + clean_step,
            clean_step,
        )
        return np.round(ticks_, 2)
    else:
        mid_point = ((y_max - y_min) / 2) + y_min
        ticks_ = [y_min, round_down(mid_point, 2), y_max]
        return np.round(ticks_, 2)


# def add_axis_break(ax, size=0.012, gap=0.018, y_pos=-0.06):
#     """Break marks on the x-axis spine."""
#     kw = dict(transform=ax.transAxes, color='black',
#               linewidth=1.0, clip_on=False, zorder=10)
#     ax.plot([-size, +size], [y_pos - size,           y_pos + size          ], **kw)
#     ax.plot([-size, +size], [y_pos - size + gap,      y_pos + size + gap    ], **kw)
#     ax.plot([0, 0],         [y_pos + size * 0.8,      y_pos + gap - size * 0.8],
#             transform=ax.transAxes, color='white',
#             linewidth=2.0, clip_on=False, zorder=9)

def add_axis_break(ax, size=0.012, gap=0.018, y_pos=-0.06, x_pos=0.01):
    """Break marks on the x-axis spine, offset to the left."""
    kw = dict(transform=ax.transAxes, color='black',
              linewidth=1.0, clip_on=False, zorder=10)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size,            y_pos + size           ], **kw)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size + gap,      y_pos + size + gap      ], **kw)
    ax.plot([x_pos, x_pos],               [y_pos + size * 0.8,      y_pos + gap - size * 0.8],
            transform=ax.transAxes, color='white',
            linewidth=2.0, clip_on=False, zorder=9)
# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_finetuning_strategies(
        all_plots,
        all_metrics=('test/ibs', 'test/concordance_index'),
        more_suptitle='',
        show_count=False,
        xticks=None,
        save_path='',
        to_plot='fet',
        num_y_ticks=6,
        axis_br_size=0.012,
        axis_br_gap=0.018,
        axis_br_y_pos=-0.02,
        y_ticks_list1=None,
        y_ticks_list2=None,
        use_auto=True,
        define_ticks=True,
        text_x=-0.2,
        text_y=0.92,
        dataset='areds',
        stylef='c.mplstyle',
):
    metric_label_dict = {'ibs':'Integrated Brier Score',
        'concordance_index': "Concordance Index",
        'brier_score_t2':"Brier Score",
        'brier_score_t4':"Brier Score",
        'brier_score_t6':"Brier Score",
        'brier_score_t8':"Brier Score",
        "brier_score_t10": "Brier Score",
        }
    with plt.style.context(stylef):

        # ── colour maps ────────────────────────────────────────────────────
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
        model_groups = {
            'dino_nako':        'in-domain FM',
            'dino_meta':        'out-of-domain FM',
            'retfound':         'in-domain FM',
            'mae_nako':         'in-domain PR',
            'simclr_INet_nako': 'in-domain PR',
            'simclr_nako':      'in-domain PR',
            'resnet_scratch':   'supervised BS',
            'resnet_INet':      'supervised BS',
        }

        group_members = {}
        for model, group in model_groups.items():
            group_members.setdefault(group, []).append(model)
        for group in group_members:
            group_members[group] = sorted(group_members[group])

        model_colors = {}
        for group, members in group_members.items():
            shades = group_shades[group]
            for i, model in enumerate(members):
                model_colors[model] = shades[i % len(shades)]

        cm_colors = {}
        for model, color in model_colors.items():
            cm_colors[model + '_lin'] = color
            cm_colors[model + '_mlp'] = color

        # ── figure: 4 independent axes ────────────────────────────────────
        # ax[0] & ax[1]  →  metric 0  (e.g. IBS,     fet / lin)
        # ax[2] & ax[3]  →  metric 1  (e.g. C-index, fet / lin)
        # No sharey — each axis gets its own y range via set_ylim so that
        # the two metrics (with completely different scales) never interfere.
        fig, ax = plt.subplots(1, 4, figsize=(12, 3), layout='tight')

        for a in ax:
            a.set_xscale('log')

        # ── plotting ───────────────────────────────────────────────────────
        for m, metrics in enumerate(all_metrics):
            for key, value_ in all_plots.items():
                value = (
                    value_
                    .groupby(['train_size'])[metrics]
                    .agg(['mean', 'std', 'count'])
                    .reset_index()
                    .rename(columns={'mean': metrics})
                )

                colour    = cm_colors.get(get_base_key(key), '#000000')
                linestyle = '.--' if '_lin_' in key else '.-'
                linewidth = 1.5

                is_fet     = to_plot in key          # True → solid (fet), False → dashed (lin)
                target_ax  = ax[m * 2 + (0 if is_fet else 1)]

                target_ax.plot(
                    value['train_size'], value[metrics],
                    linestyle,
                    color=colour, linewidth=linewidth, markersize=4,
                )

                if show_count:
                    for x_val, y_val, c_val in zip(
                            value['train_size'], value[metrics], value['count']):
                        target_ax.text(x_val, y_val, str(c_val), fontsize=5)

            # ── axis formatting (once per metric, after all keys plotted) ──
            # Compute limits separately for each subplot in the pair so
            # the fet (solid) and lin (dashed) y-ranges match correctly.
            fet_plots = {k: v for k, v in all_plots.items() if to_plot in k}
            lin_plots = {k: v for k, v in all_plots.items() if to_plot not in k}

            # Use the union of both subsets for a shared range within the pair

            if dataset == 'areds':
                x_ticks = [100, 1000, 10000, 35000]
            else:
                x_ticks = [500, 1000, 10000, 35000]
            current_font = plt.rcParams.get('font.family', 'arial')


            print('metrics', metrics)
            if metrics == 'test/ibs':
                y_min0, y_max0, _ = get_axis_limits(all_plots, metrics)
                # if m == 0:
                yticks0 = get_yticks(
                y_min0, y_max0,
                use_auto=False, define_ticks=define_ticks,
                n_ticks=num_y_ticks, ticks=y_ticks_list1,
            )
             
            else:
                y_min1, y_max1, _ = get_axis_limits(all_plots, metrics)
                yticks1 = get_yticks(
                y_min1, y_max1,
                use_auto=False, define_ticks=define_ticks,
                n_ticks=num_y_ticks, ticks=y_ticks_list2,
            )
            

            for a in [ax[m * 2], ax[m * 2 + 1]]:
                if a is not ax[m * 2]:  # hide y ticks on right axis
                    a.tick_params(labelleft=False, left = False)
                if a is ax[m * 2]:
                    metric_label = metrics.split('/')[1].upper() if '/' in metrics else metrics.upper()
                    metric_label = metric_label.lower()

                    a.set_ylabel(f'{metric_label_dict[metric_label]}', fontsize=8, 
                                fontfamily=current_font,
                                x=0.045) 
                if metrics == 'test/ibs':
                    a.set_yticks(yticks0)
                    a.set_ylim(yticks0[0], yticks0[-1])
                    
                else:
                    a.set_yticks(yticks1)
                    a.set_ylim(yticks1[0], yticks1[-1])

            #     # x-axis
                a.set_xticks([], minor=True)
                a.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, _: (
                        f'$10^{{{int(np.log10(x))}}}$'
                        if x in [100, 1000, 10000]
                        else f'${int(x / 1000)}$k'
                    )
                ))
                a.spines['bottom'].set_bounds(x_ticks[0], x_ticks[-1])

                sns.despine(ax=a, offset={'left': 4, 'bottom':5}, trim=False,)
            #     a.tick_params(labelsize=5)
                add_axis_break(
                    a,
                    size=axis_br_size,
                    gap=axis_br_gap,
                    y_pos=axis_br_y_pos,
                )

                 
                fig.supxlabel('Train size', fontsize=8, fontfamily=current_font,  x=0.4, y=0.01 )
                
        # ── optional legend on last axis ───────────────────────────────────
        # Uncomment the next two lines to add the publication legend:

        make_publication_legend(ax[3], cm_colors, model_groups,
                                group_colors, outside=True)
        for a, label in zip(ax.flat, 'abcdefgh'):
            a.text(text_x, text_y, f'({label})', transform=a.transAxes,
            fontsize=6, fontweight='bold',
            ha='left', va='top', clip_on=False)
        
                # <-- pull ylabel closer to axes

        fig.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(bottom=0.15)
        plt.suptitle(more_suptitle,
                    fontsize=7, 
                    fontfamily=current_font,
                    x = 0.45,
                    y = 1.02,
                    fontweight = 'bold'
                    )
        plt.show()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
        
    
def get_remnants(df_, survival_head = 'mlp', freeze_encoder = 0 , data = 'eyepacs'):
  df_ = df_[(df_['survival_head'] == survival_head) & (df_['freeze_encoder'] == freeze_encoder) ]
  if data == 'eyepacs':
    expected_train_size = [500, 1000, 10000, 20000, 30000, 32500]
  else:
    expected_train_size = [100, 500, 1000, 10000, 20000, 30000, 32500]

  expected_train_size = [float(i) for i in expected_train_size]

  models = df_['summary'].unique()
  expected = pd.DataFrame(
      list(product(models, expected_train_size)),
      columns=['summary', 'train_size']
  )
  run_counts = df_.groupby(['summary', 'train_size']).size().reset_index(name='run_count')

  # Merge to find missing or incomplete
  result = expected.merge(run_counts, on=['summary', 'train_size'], how='left')
  result['run_count'] = result['run_count'].fillna(0).astype(int)

  missing_or_incomplete = result[result['run_count'] < 5]
  return missing_or_incomplete



