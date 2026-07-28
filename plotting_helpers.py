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
