import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math
import string

PANEL_LABELS = 'bold'
FONTNAME  = 'Arial'
LEGEND_FONTSIZE = 9

MODEL_GROUPS = {
                'dino_nako':        ['in-domain FM', '#000000'],
                'dino_meta':        ['out-of-domain FM', '#05771D'],
                'retfound':         ['in-domain FM',  '#90A4AE'],
                'nako_mae':         ['in-domain PR', '#6A1B9A'],
                'nako_mae_scr':     ['in-domain PR', '#9C74B5'],
                'simclr_INet_nako': ['in-domain PR', '#D81B60'],
                'simclr_nako':      ['in-domain PR', '#F3D0DC'],
                'resnet_scratch':   ['supervised BS', '#1565C0'],
                'resnet_INet':      ['supervised BS', '#42A5F5'],
            }
NEW_NAMES = {
        'dino_nako':        'DINOv2-NAKO',
        'dino_meta':        'DINOv2-LVD',
        'retfound':         'RETFound',
        'resnet_INet':      'ResNet-ImageNet',
        'nako_mae':         'MAE-ImageNet-NAKO',
        'nako_mae_scr':     'MAE-NAKO',
        'simclr_nako':      'SimCLR-NAKO',
        'simclr_INet_nako': 'SimCLR-ImageNet-NAKO',
        'resnet_scratch':   'ResNet-Scratch',
    }

def get_color_legend():

    
    group_colors = {
            'supervised BS':  '#1565C0',
            'out-of-domain FM': '#05771D',
            'in-domain PR':   '#6A1B9A',
            'in-domain FM':   '#050606',
        }
      
        
    group_members, model_colors, color_groups =  {}, {}, {}
    for model, group in MODEL_GROUPS.items():
        group_members.setdefault(group[0], []).append(model)
        model_colors[model] = group[1]
        color_groups[model] = group[1]
      

    cm_colors = {}
    for model, color in model_colors.items():
        cm_colors[model + '_lin'] = color
        cm_colors[model + '_mlp'] = color

    return group_colors, cm_colors, color_groups



def get_base_key(plot_key):
    """'retfound_mlp_fet' -> 'retfound_mlp'"""
    return plot_key.rsplit('_', 1)[0]
def add_axis_break(ax, size=0.012, gap=0.018, y_pos=-0.06, x_pos=0.01, spine_x=0.0):
    
    kw = dict(transform=ax.transAxes, color='black',
              linewidth=1.0, clip_on=False, zorder=10)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size,       y_pos + size       ], **kw)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size + gap, y_pos + size + gap ], **kw)

    ax.plot([spine_x, spine_x], [y_pos + size * 0.8, y_pos + gap - size * 0.8],
            transform=ax.transAxes, color='white',
            linewidth=3.0, clip_on=False, zorder=9) 

    
def get_axis_limits(all_plots, metrics, x_col='train_size', margin=0.01):
    all_y = []
    all_x = set()

    for value_ in all_plots.values():
        agg = value_.groupby(x_col)[metrics].mean()
        all_y.extend(agg.values)
        all_x.update(agg.index.tolist())

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

def make_publication_legend_columns(cm_colors, model_groups,
                                    group_colors, present_models = None):
    
    group_members = {}
    for model, group in sorted(model_groups.items()):
        if present_models is not None and model not in present_models:
            continue
        group_members.setdefault(group[0], [])
        group_members[group[0]].append(model)

    group_order = ['in-domain FM', 'out-of-domain FM',
                    'in-domain PR', 'supervised BS']

    group_display = {
        'in-domain FM':     'In-domain FM',
        'out-of-domain FM': 'Out-of-domain FM',
        'in-domain PR':     'In-domain PR',
        'supervised BS':    'Supervised BS',
    }

    
    columns = []
    for group in group_order:
        members = sorted(group_members.get(group, []))

        handles = [mpatches.Patch(color=group_colors[group], alpha=0.25)]
        labels  = [group_display[group]]
        raw     = [group_display[group]]  

        for model in members:
            color = cm_colors.get(model + '_lin', '#000000')
            handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=1.2))
            display = model

            display = model
            for old, new in sorted(NEW_NAMES.items(), key=lambda kv: -len(kv[0])):
                if old in display:
                    display = display.replace(old, new)
                    break   
   
            labels.append(display)
            raw.append(None)   # not a header

        columns.append((handles, labels, raw))
    return columns


def plot_finetuning_strategies_all(
        rows,                                   
        row_titles=None,                        
        all_metrics=('test/ibs', 'test/concordance_index'),
        more_suptitle='',
        show_count=False,
        xticks=None,
        save_path='',
        to_plot='fet',
        num_y_ticks=6,
        axis_br_size=0.012,
        axis_br_gap=0.018,
        axis_br_x_pos=-0.005,
        axis_br_y_pos=-0.02,
        y_ticks_list1=None,
        y_ticks_list2=None,
        use_auto=True,
        define_ticks=True,
        text_x=-0.2,
        text_y=1.065,
        dataset='areds',
        stylef='c.mplstyle',
        spine_linewidth=1.,
        tick_width=1.,
        tick_length=5,
        fontsize=6,
        remove_mae_scr = True
    ):

    metric_label_dict = {
        'ibs': 'Integrated Brier Score',
        'concordance_index': "Concordance Index",
        'brier_score_t2': "Brier Score",
        'brier_score_t4': "Brier Score",
        'brier_score_t6': "Brier Score",
        'brier_score_t8': "Brier Score",
        "brier_score_t10": "Brier Score",
    }
    n_rows = len(rows)  

    with plt.style.context(stylef):
        
        present_models = set()
        for all_plots in rows:
            for key in all_plots.keys():
                m= "_".join(get_base_key(key).split('_')[:-1])
                present_models.add(m)
       
        group_colors, cm_colors, _ = get_color_legend()

        width_pt = 451
        width_in = width_pt / 72.27
        golden_ratio = (5**0.5 - 1) / 2
        height_per_row = 2.6  

        fig, ax = plt.subplots(n_rows, 4, figsize=(width_in, height_per_row * n_rows),
                                layout='tight', squeeze=False)

        for row_axes in ax:
            for a in row_axes:
                a.set_xscale('log')

        for r, all_plots in enumerate(rows):                
            row_ax = ax[r]                                   

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
                    linewidth = 1.
                    is_fet    = to_plot in key
                    target_ax = row_ax[m * 2 + (1 if is_fet else 0)]   

                    target_ax.plot(
                        value['train_size'], value[metrics],
                        linestyle,
                        color=colour, linewidth=linewidth, markersize=3,
                    )

                    if show_count:
                        for x_val, y_val, c_val in zip(
                                value['train_size'], value[metrics], value['count']):
                            target_ax.text(x_val, y_val, str(c_val), fontsize=fontsize)

                if metrics == 'test/ibs':
                    y_min0, y_max0, _ = get_axis_limits(all_plots, metrics)
                    yticks0 = get_yticks(
                        y_min0, y_max0, use_auto=False, define_ticks=define_ticks,
                        n_ticks=num_y_ticks, ticks=y_ticks_list1,
                    )
                else:
                    y_min1, y_max1, _ = get_axis_limits(all_plots, metrics)
                    yticks1 = get_yticks(
                        y_min1, y_max1, use_auto=False, define_ticks=define_ticks,
                        n_ticks=num_y_ticks, ticks=y_ticks_list2,
                    )

                for a in [row_ax[m * 2], row_ax[m * 2 + 1]]:
                    if a is row_ax[m * 2]:
                        metric_label = metrics.split('/')[1].upper() if '/' in metrics else metrics.upper()
                        metric_label = metric_label.lower()
                        a.set_ylabel(f'{metric_label_dict[metric_label]}', fontsize=fontsize, x=0.045)

                    if metrics == 'test/ibs':
                        a.set_yticks(yticks0)
                        a.set_ylim(yticks0[0], yticks0[-1])
                    else:
                        a.set_yticks(yticks1)
                        a.set_ylim(yticks1[0], 1.0)

                    all_positions = [1e2, 1e3, 1e4, 1e5]
                    a.set_xticks(all_positions)
                    labels = [r'$10^2$', '', '', r'$10^5$']
                    a.set_xticklabels(labels)
                    a.tick_params(axis='both', labelsize=fontsize, width=tick_width, 
                                  length=tick_length)
                    a.tick_params(axis='x', which='minor', labelbottom=False)

                    a.set_xlim(80, 1.2e5)
                    sns.despine(ax=a, offset={'left': 3, 'bottom': 3}, trim=False)
                    for spine in ['left', 'bottom']:
                        a.spines[spine].set_linewidth(spine_linewidth)

            if row_titles is not None and r < len(row_titles):
                row_ax[0].annotate(
                    row_titles[r], xy=(-0.35, 0.5), xycoords='axes fraction',
                    fontsize=fontsize + 1, fontweight='bold',
                    ha='right', va='center', rotation=90
                )
        proxy_mlp = Line2D([0], [0], linestyle='-',  color='gray', linewidth=2, label='MLP head')
        proxy_lin = Line2D([0], [0], linestyle='--', color='gray', linewidth=2, label='Linear head')
        group_columns = make_publication_legend_columns(cm_colors, MODEL_GROUPS, 
                                                         group_colors, present_models)

        fm_in_handles,  fm_in_labels,  fm_in_raw  = group_columns[0]
        fm_out_handles, fm_out_labels, fm_out_raw = group_columns[1]
        merged_fm_column = (
            fm_in_handles + fm_out_handles,
            fm_in_labels  + fm_out_labels,
            fm_in_raw     + fm_out_raw,
        )
        group_columns = [merged_fm_column] + group_columns[2:]

        head_column = (
            [Line2D([0], [0], color='none'), proxy_mlp, proxy_lin],
            ['Head type', 'MLP head', 'Linear head'],
            ['Head type', None, None],
        )

        all_columns = [head_column] + group_columns
        max_rows_legend = max(len(h) for h, l, r_ in all_columns)

        flat_handles, flat_labels, flat_raw = [], [], []
        for handles, labels, raw in all_columns:
            pad = max_rows_legend - len(handles)
            handles = handles + [Line2D([0], [0], color='none')] * pad
            labels  = labels  + [''] * pad
            raw     = raw     + [None] * pad
            flat_handles.extend(handles)
            flat_labels.extend(labels)
            flat_raw.extend(raw)

        legend_y = 0.2

        combined_legend = fig.legend(
            handles=flat_handles,
            labels=flat_labels,
            loc='upper center',
            bbox_to_anchor=(0.51, legend_y),
            ncol=len(all_columns),
            fontsize=LEGEND_FONTSIZE,
            edgecolor='#cccccc',
            frameon=False,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        for text, raw_label in zip(combined_legend.get_texts(), flat_raw):
            if raw_label is not None:
                text.set_fontweight('bold')
                

        panel_labels = string.ascii_uppercase[: n_rows * 4]
        for a, label in zip(ax.flat, panel_labels):
            a.text(text_x, text_y, f'{label}',
                   transform=a.transAxes,
                   fontsize=10, fontweight=PANEL_LABELS,
                   ha='left', va='top', clip_on=False,
                   fontname = FONTNAME)

        fig.supxlabel('Longitudinal training set size (#images)', fontsize=fontsize, 
                                               x=0.5, y=0.21)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15 * n_rows if n_rows > 1 else 0.15)  
        extra_gap = 0.01
        for row_axes in ax:
            for i, a in enumerate(row_axes):
                pos = a.get_position()
                shift = extra_gap if i >= 2 else 0
                a.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

        plt.show()
        if save_path:
            fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')

        return fig, ax