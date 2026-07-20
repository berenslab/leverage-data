
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brokenaxes import brokenaxes
from matplotlib.legend_handler import HandlerLine2D

def draw_curves(all_img_to_plot, curves_dict,
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

    fig = plt.figure(figsize=(len(all_img_to_plot) * 3, len(all_img_to_plot)))
    gs = gridspec.GridSpec(1, len(all_img_to_plot), figure=fig, wspace=0.3)

    y_min = min(min(curve) for curves in curves_dict.values() for curve in curves) - 0.02
    y_max = 1
    x = [1, 2, 3, 4, 5]

    # ── Derive conversion marker position ─────────────────────────────────────
    grades = all_img_to_plot[1]   # e.g. [7.0, 8.0, 9.0, 9.0, 11.0]
    visit_number = all_img_to_plot[-1]

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

            img_id = all_img_to_plot[2][0]
            bax.set_title(f"ID {img_id} V_no {visit_number[i]} Label {int(grades[i])}", fontsize=10)
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

def get_model_curves(df, model_name, img_to_plot, CHKPT_DIR):
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


def get_all_curves(all_images_to_plot0, model_list, dfl_ssl, CHKPT_DIR):
    print(all_images_to_plot0)
    curves_dict = {}
    for img_to_plot in all_images_to_plot0:
        for m in model_list:
            surv_curv = get_model_curves(dfl_ssl, m, img_to_plot, CHKPT_DIR)
            if m not in curves_dict:
                curves_dict[m]=[]
            curves_dict[m].append(surv_curv)
    print('len of curves_dict', curves_dict)
    return curves_dict