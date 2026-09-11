# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import numpy as np
import math
import os
# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
def make_publication_legend_columns(cm_colors, model_groups, group_colors):
    """
    Returns one (handles, labels, raw_labels) tuple per group,
    each representing a legend COLUMN with the group header on top.
    """
    group_members = {}
    for model, group in sorted(model_groups.items()):
        group_members.setdefault(group, [])
        group_members[group].append(model)

    group_order = ['in-domain FM', 'out-of-domain FM', 'in-domain PR', 'supervised BS']

    group_display = {
        'in-domain FM':     'In-domain FM',
        'out-of-domain FM': 'Out-of-domain FM',
        'in-domain PR':     'In-domain PR',
        'supervised BS':    'Supervised BS',
    }

    new_names = {
        'dino_nako':        'DINOv2 NAKO',
        'dino_meta':        'DINOv2 LVD',
        'retfound':         'RETFound',
        'resnet_INet':      'ResNet INet',
        'mae_nako':         'MAE NAKO',
        'simclr_nako':      'SimCLR NAKO',
        'simclr_INet_nako': 'SimCLR INet NAKO',
        'resnet_scratch':   'ResNet Scratch',
    }

    columns = []
    for group in group_order:
        members = sorted(group_members.get(group, []))

        handles = [mpatches.Patch(color=group_colors[group], alpha=0.25)]
        labels  = [group_display[group]]
        raw     = [group_display[group]]   # non-None marks "this is a header -> bold it"

        for model in members:
            color = cm_colors.get(model + '_lin', '#000000')
            handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=1.2))
            display = model
            for old, new in new_names.items():
                if old in display:
                    display = display.replace(old, new)
            labels.append(display)
            raw.append(None)   # not a header

        columns.append((handles, labels, raw))
    return columns

def make_publication_legend(ax, cm_colors, model_groups, group_colors, 
                            outside=True, bbox_to_anchor=(1.05, 0.2),
                            return_handles=False):          # <-- new param
    handles = []
    labels  = []

    group_members = {}
    for model, group in sorted(model_groups.items()):
        group_members.setdefault(group, [])
        group_members[group].append(model)

    group_order = ['in-domain FM', 'out-of-domain FM', 'in-domain PR', 'supervised BS']

    for group in group_order:
        members = sorted(group_members.get(group, []))
        handles.append(mpatches.Patch(color=group_colors[group], alpha=0.25))
        if group == 'in-domain FM':
            labels.append('In-domain FM')
        elif group == 'out-of-domain FM':
            labels.append('Out-of-domain FM')
        elif group == 'in-domain PR':
            labels.append('In-domain PR')
        elif group == 'supervised BS':
            labels.append('Supervised BS')
        for model in members:
            color = cm_colors.get(model + '_lin', '#000000')
            handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=1.2))
            labels.append(f'  {model}')
        handles.append(Line2D([0], [0], color='none'))
        labels.append('')

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

    # apply renaming to labels up front, regardless of return path
    display_labels = []
    for label in labels:
        new_label = label
        for old, new in new_names.items():
            if old in new_label:
                new_label = new_label.replace(old, new)
        display_labels.append(new_label)

    if return_handles:
        return handles, display_labels, labels   # labels (raw) needed to know which to bold

    legend_kwargs = dict(
        handles=handles,
        labels=display_labels,
        frameon=False,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=10,
        handlelength=2.0,
        handleheight=1.0,
        handletextpad=0.5,
        borderpad=0.8,
        labelspacing=0.3,
    )

    if outside:
        leg = ax.legend(**legend_kwargs,
                        bbox_to_anchor=bbox_to_anchor,
                        bbox_transform=ax.get_figure().transFigure,
                        loc='upper left',
                        borderaxespad=0.)
    else:
        leg = ax.legend(**legend_kwargs, loc='best')

    for text, label in zip(leg.get_texts(), labels):
        if not label.startswith('  ') and label != '':
            text.set_fontweight('bold')

    return leg
# def make_publication_legend(ax, cm_colors, model_groups, group_colors, 
#                             outside=True, bbox_to_anchor = (1.05, 0.2)):
#     """
#     Two-level legend: group label (colored patch) + indented model entries.
#     outside=True places it to the right of the axes.
#     """
#     handles = []
#     labels  = []

#     group_members = {}
#     for model, group in sorted(model_groups.items()):
#         group_members.setdefault(group, [])
#         group_members[group].append(model)

#     group_order = ['in-domain FM', 'out-of-domain FM', 'in-domain PR', 'supervised BS']

    
#     # cpitalize group names for display
    
#     for group in group_order:
#         members = sorted(group_members.get(group, []))

#         # group header: filled patch
#         handles.append(mpatches.Patch(color=group_colors[group], alpha=0.25))
#         if group == 'in-domain FM':
            
#             labels.append('In-domain FM')
#         elif group == 'out-of-domain FM':
#             labels.append('Out-of-domain FM')
#         elif group == 'in-domain PR':
#             labels.append('In-domain PR')
#         elif group == 'supervised BS':
#             labels.append('Supervised BS')
#         # model entries: solid line
#         for model in members:
#             color = cm_colors.get(model + '_lin', '#000000')
#             handles.append(Line2D([0], [0], color=color, linestyle='-',
#                                   linewidth=1.2))
#             labels.append(f'  {model}')

#         # blank spacer between groups
#         handles.append(Line2D([0], [0], color='none'))
#         labels.append('')

#     legend_kwargs = dict(
#         handles=handles,
#         labels=labels,
#         frameon=False,
#         framealpha=0.95,
#         edgecolor='#cccccc',
#         fontsize=10,
#         handlelength=2.0,
#         handleheight=1.0,
#         handletextpad=0.5,
#         borderpad=0.8,
#         labelspacing=0.3,
#     )

#     if outside:
#         leg = ax.legend(**legend_kwargs,
#                         bbox_to_anchor=bbox_to_anchor,
#                         bbox_transform=ax.get_figure().transFigure,   # <-- ADD THIS
#                         loc='upper left',
#                         borderaxespad=0.)
#     else:
#         leg = ax.legend(**legend_kwargs, loc='best')

#     new_names = {
#         'dino_nako':       'DINOv2 NAKO',
#         'dino_meta':       'DINOv2 LVD',
#         'retfound':        'RETFound',
#         'resnet_INet':     'ResNet INet',
#         'mae_nako':        'MAE NAKO',
#         'simclr_nako':     'SimCLR NAKO',
#         'simclr_INet_nako':'SimCLR INet NAKO',
#         'resnet_scratch':  'ResNet Scratch',
#     }

#     for text, label in zip(leg.get_texts(), labels):
#         if not label.startswith('  ') and label != '':
#             text.set_fontweight('bold')
#         new_label = label
#         for old, new in new_names.items():
#             if old in new_label:
#                 new_label = new_label.replace(old, new)
#         text.set_text(new_label)

#     return leg



def get_base_key(plot_key):
    """'retfound_mlp_fet' -> 'retfound_mlp'"""
    return plot_key.rsplit('_', 1)[0]
def add_axis_break(ax, size=0.012, gap=0.018, y_pos=-0.06, x_pos=0.01, spine_x=0.0):
    """Break marks on the x-axis spine, offset to the left.
    x_pos: horizontal position of the diagonal break marks (adjustable).
    spine_x: actual x-position of the spine to erase (should match sns.despine offset).
    """
    kw = dict(transform=ax.transAxes, color='black',
              linewidth=1.0, clip_on=False, zorder=10)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size,       y_pos + size       ], **kw)
    ax.plot([x_pos - size, x_pos + size], [y_pos - size + gap, y_pos + size + gap ], **kw)

    # eraser stays locked to the true spine x-position, not x_pos
    ax.plot([spine_x, spine_x], [y_pos + size * 0.8, y_pos + gap - size * 0.8],
            transform=ax.transAxes, color='white',
            linewidth=3.0, clip_on=False, zorder=9)   # increased linewidth for full coverage

# def add_axis_break(ax, size=0.012, gap=0.018, y_pos=-0.06, x_pos=0.01, spine_x=0.0):
#     """Break marks on the x-axis spine, offset to the left.
#     x_pos: horizontal position of the diagonal break marks (adjustable).
#     spine_x: actual x-position of the spine to erase (should match sns.despine offset).
#     """
#     kw = dict(transform=ax.transAxes, color='black',
#               linewidth=1.0, clip_on=False, zorder=10)
#     ax.plot([x_pos - size, x_pos + size], [y_pos - size,       y_pos + size       ], **kw)
#     ax.plot([x_pos - size, x_pos + size], [y_pos - size + gap, y_pos + size + gap ], **kw)

#     # eraser stays locked to the true spine x-position, not x_pos
#     ax.plot([spine_x, spine_x], [y_pos + size * 0.8, y_pos + gap - size * 0.8],
#             transform=ax.transAxes, color='white',
#             linewidth=3.0, clip_on=False, zorder=9)   # increased linewidth for full coverage
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


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brokenaxes import brokenaxes
from matplotlib.legend_handler import HandlerLine2D

def get_all_curves(all_images_to_plot0, model_list, dfl_ssl, CHKPT_DIR):
    # print(all_images_to_plot0)
    # all_images_to_plot0 = all_images_to_plot0['image_path']
    curves_dict = {}
    for img_to_plot in all_images_to_plot0:
        for m in model_list:
            surv_curv = get_model_curves(dfl_ssl, m, img_to_plot, CHKPT_DIR)
            if m not in curves_dict:
                curves_dict[m]=[]
            curves_dict[m].append(surv_curv)
    print('len of curves_dict', curves_dict)
    return curves_dict

def get_model_curves(df, model_name, img_to_plot, CHKPT_DIR):
    print(df.head(3))
    print(df.columns)
    row = df[(df['model']==model_name)]
    print('img_to_plot', img_to_plot)
    exp_best = row['best_result'].values[0]
    full_path = os.path.join(CHKPT_DIR, exp_best)
    exp_df = pd.read_csv(os.path.join(full_path, 'test_survival_curves.csv' ))
    image_curve = img_to_plot
    print('image_curve', image_curve)
    image_curve = image_curve.split('/')[-1]
    print('image_curve', image_curve)

    h = exp_df[exp_df['image_name'] == image_curve ]
    print('h', h)
    s = h['survival_curve'].values[0].strip().strip('[]')
    print('s', s)
    surv_curv = [float(x) for x in s.split()]
    print('surv_curv',surv_curv)
    return surv_curv

def draw_curves(data_to_plot, curves_dict,
                first_conversion_idx=None,
                 marker_panel_idx=None, save_path=None, stylef=None):
    """
    all_img_to_plot : [images, labels, [img_id]]
                      labels = list of diagnosis_amd_grade per visit
    curves_dict     : {model_name: [curve_visit_0, curve_visit_1, ...]}

    Conversion marker: a single grey downward triangle on the x-axis at the
    conversion year, shown only in the penultimate panel (last pre-conversion visit).
    """
    color_groups = {
        'dino_nako': '#000000',
        'dino_meta': '#05771D',
        'retfound':  '#90A4AE',  # fixed typo
        'nako_mae':  '#6A1B9A',
        'simclr_inet_nako': '#D81B60',
        'simclr_nako': '#F3D0DC',
        'scratch': '#1565C0',
        'resnet_imagenet': '#42A5F5',
    }
    new_names = {
        'dino_nako': 'DINOv2 NAKO',
        'dino_meta': 'DINOv2 LVD',
        'retfound': 'RETFound',
        'resnet_imagenet': 'ResNet INet',
        'nako_mae': "MAE NAKO",
        'simclr_nako': 'SimCLR NAKO',
        'simclr_inet_nako': 'SimCLR INet NAKO',
        'scratch': "ResNet Scratch"
    }


    all_img_to_plot = data_to_plot#['image_path']
    print('all_img_to_plot',all_img_to_plot)

    n_images = len(all_img_to_plot)
    fig = plt.figure(figsize=(n_images * 3, n_images))
    gs = gridspec.GridSpec(1, n_images, figure=fig, wspace=0.3)

    y_min = min(min(curve) for curves in curves_dict.values() for curve in curves) - 0.02
    y_max = 1
    x = [1, 2, 3, 4, 5]

    # ── Derive conversion marker position ─────────────────────────────────────
    grades =data_to_plot['diagnosis_amd_grade']  # e.g. [7.0, 8.0, 9.0, 9.0, 11.0]
    print(grades)
    visit_number = data_to_plot['visit_number']
    train_size = data_to_plot['train_size']   # <-- added
    # Always initialize so these names exist regardless of which branch runs
    marker_x = None

    if any(g in [10, 11, 12] for g in grades):

        if first_conversion_idx is None:
            first_conversion_idx = next(
                (i for i, g in enumerate(grades) if g >= 10), None
            )

            if first_conversion_idx is not None and first_conversion_idx > 0:
                marker_panel_idx = first_conversion_idx - 1   # penultimate panel
                marker_x         = first_conversion_idx + 1   # 1-based conversion year
            else:
                marker_panel_idx = None
                marker_x         = None
        else:
            marker_x = first_conversion_idx
            # marker_panel_idx = first_conversion_idx - 2

        print(f'marker x is {marker_x} marker_panel_idx is {marker_panel_idx}')
    else:
        # No conversion grade present in this sample at all
        marker_panel_idx = None
        marker_x = None
    # ──────────────────────────────────────────────────────────────────────────

    with plt.style.context(stylef):
        for i, img in enumerate(all_img_to_plot):
            bax = brokenaxes(
                xlims=((-0.1, 0.002), (0.9, max(x) + 0.1)),
                hspace=0.01,
                d=0.001,
                tilt=45,
                subplot_spec=gs[i],
                fig=fig
            )

            for model, curves in curves_dict.items():
                curve_to_plot = curves[i]
                bax.plot(
                    range(1, len(curve_to_plot) + 1),
                    curve_to_plot,
                    label=new_names[model],
                    color=color_groups[model],
                    linewidth=1.5
                )

            # ── Single conversion marker on the x-axis ────────────────────────
            if i == marker_panel_idx and marker_x is not None:
                # bax.axs[1] is the main (right) axes where the curves are drawn
                bax.axs[1].plot(
                    marker_x,
                    0.02,              # sit right on the x-axis
                    marker='v',
                    markersize=10,
                    color='grey',
                    linestyle='none',
                    clip_on=False,      # don't clip if right at the edge
                    zorder=5
                )
            # ──────────────────────────────────────────────────────────────────

            img_id = all_img_to_plot[0].split('/')[0]
            bax.set_title(
                f"ID {img_id} V_no {visit_number[i]} Label {int(grades[i])} TS {train_size[i]}",
                 fontsize=10)
            bax.set_ylim(y_min, y_max)
            if i == 0:
                bax.set_ylabel('S(t)', fontsize=10)       # y-label on the leftmost panel only
            if i == len(all_img_to_plot[0]) // 2:
                bax.set_xlabel('Year', fontsize=10)
            bax.axs[0].set_xticks([])
            bax.axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
            for ax in bax.axs:
                ax.tick_params(axis='both', labelsize=10)
                ax.set_yticks([0, 0.5, 1.0])

            if i == len(all_img_to_plot[0]) - 1:
                handles, labels = [], []
                for model in curves_dict:
                    handles.append(plt.Line2D([0], [0],
                                               color=color_groups[model],
                                               linewidth=1.5))
                    labels.append(new_names[model])

                if marker_panel_idx is not None and marker_x is not None:
                    handles.append(plt.Line2D([0], [0],
                                               marker='v', color='w',
                                               markerfacecolor='grey',
                                               markersize=8,
                                               linestyle='none'))
                    labels.append('Conversion')

                leg = bax.legend(
                    handles, labels,
                    loc='lower left',
                    bbox_to_anchor=(1, 0.2),
                    labelspacing=1.5,
                    fontsize=8,
                    handler_map={plt.Line2D: HandlerLine2D(numpoints=3)}
                )
                for line in leg.get_lines():
                    line.set_linewidth(1.5)

    if save_path is not None:
        plt.savefig(f'{save_path}/surv_curves_{img_id}.pdf')
    plt.show()
