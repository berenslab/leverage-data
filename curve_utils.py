
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brokenaxes import brokenaxes
from matplotlib.legend_handler import HandlerLine2D
rng = np.random.default_rng(seed=42)

color_groups = {
    'dino_nako': '#000000',
    'dino_meta': '#05771D',
    'retfound':  '#90A4AE',
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


def draw_curves_grid_simple(full_curves_dict, info_dict=None, ids=None,
                             n_rows=4, n_cols=4, save_path=None, stylef=None):
    n_patients = len(next(iter(full_curves_dict.values())))  # length of any model's curve list

    y_min = min(min(curve) for curves in full_curves_dict.values() for curve in curves) - 0.02
    y_max = 1

    with plt.style.context(stylef):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), sharey=True)
        axes = axes.flatten()

        for p in range(n_patients):
            ax = axes[p]
            for model, curves in full_curves_dict.items():
                curve = curves[p]
                ax.plot(range(1, len(curve) + 1), curve,
                         label=new_names.get(model, model),
                         color=color_groups.get(model),
                         linewidth=1.5)

            title = f"Patient {ids[p]}" if ids is not None else f"Config {p}"
            if info_dict is not None:
                cfg = info_dict[p]
                title += f"\nfreeze={cfg['freeze_encoder']} n={cfg['train_size']} evt={cfg['has_event']}"
            ax.set_title(title, fontsize=8)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis='both', labelsize=7)

            if p % n_cols == 0:
                ax.set_ylabel('S(t)', fontsize=9)
            if p >= n_cols * (n_rows - 1):
                ax.set_xlabel('Year', fontsize=9)

        # turn off any unused axes if n_patients < n_rows*n_cols
        for j in range(n_patients, len(axes)):
            axes[j].axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(f'{save_path}/surv_curves_grid.pdf', bbox_inches='tight')
    plt.show()

def pick_same_index_across_lists(row):
    # print('--------------------------------------')
    # print()
    # print(row)
    # print('--------------------------------------')
    # print()
    list_lengths = [len(v) for v in row if isinstance(v, list)]
    if not list_lengths:
        return row

    n = list_lengths[0]
    chosen_idx = n -1 #rng.integers(0, n)

    return row.apply(lambda v: [v[chosen_idx]] if isinstance(v, list) else v)
def draw_curves_grid(data_to_plot, curves_dict,
                 first_conversion_idx=None,
                 marker_panel_idx=None,
                 fig=None, subplot_spec=None,   # NEW
                 save_path=None, stylef=None,
                 show_legend=True):             # NEW: let caller suppress repeated legends

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

    all_img_to_plot = data_to_plot['image_path']
    n_images = len(all_img_to_plot)

    standalone = fig is None  # NEW: are we making our own figure, or drawing into someone else's?

    if standalone:
        print('in standalone')
        fig = plt.figure(figsize=(n_images * 3, n_images))
        outer_gs = gridspec.GridSpec(1, n_images, figure=fig, wspace=0.3)
        cells = [outer_gs[i] for i in range(n_images)]
    else:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_images, subplot_spec=subplot_spec, wspace=0.3
        )
        cells = [inner_gs[i] for i in range(n_images)]

    y_min = min(min(curve) for curves in curves_dict.values() for curve in curves) - 0.02
    y_max = 1
    x = [1, 2, 3, 4, 5]

    grades = data_to_plot['diagnosis_amd_grade']
    visit_number = data_to_plot['visit_number']
    marker_x = None
    marker_panel_idx_local = marker_panel_idx

    if any(g in [10, 11, 12] for g in grades):
        if first_conversion_idx is None:
            fc_idx = next((i for i, g in enumerate(grades) if g >= 10), None)
            if fc_idx is not None and fc_idx > 0:
                marker_panel_idx_local = fc_idx - 1
                marker_x = fc_idx + 1
        else:
            marker_x = first_conversion_idx
    # ------------------------------------------------------------------

    ctx = plt.style.context(stylef) if standalone else contextlib.nullcontext()
    with ctx:
        for i, img in enumerate(all_img_to_plot):
            bax = brokenaxes(
                xlims=((-0.1, 0.002), (0.9, max(x) + 0.1)),
                hspace=0.01, d=0.001, tilt=45,
                subplot_spec=cells[i],
                fig=fig
            )

            for model, curves in curves_dict.items():
                curve_to_plot = curves[i]
                bax.plot(range(1, len(curve_to_plot) + 1), curve_to_plot,
                          label=new_names[model], color=color_groups[model], linewidth=1.5)

            if i == marker_panel_idx_local and marker_x is not None:
                bax.axs[1].plot(marker_x, 0.02, marker='v', markersize=10,
                                 color='grey', linestyle='none', clip_on=False, zorder=5)

            img_id = all_img_to_plot[0].split('/')[0]
            bax.set_title(f"ID {img_id} V_no {visit_number[i]} Label {int(grades[i])}", fontsize=8)
            bax.set_ylim(y_min, y_max)
            if i == 0:
                bax.set_ylabel('S(t)', fontsize=8)
            bax.axs[0].set_xticks([])
            bax.axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
            for ax in bax.axs:
                ax.tick_params(axis='both', labelsize=7)
                ax.set_yticks([0, 0.5, 1.0])

            if show_legend and i == n_images - 1:
                handles, labels = [], []
                for model in curves_dict:
                    handles.append(plt.Line2D([0], [0], color=color_groups[model], linewidth=1.5))
                    labels.append(new_names[model])
                if marker_panel_idx_local is not None and marker_x is not None:
                    handles.append(plt.Line2D([0], [0], marker='v', color='w',
                                               markerfacecolor='grey', markersize=8, linestyle='none'))
                    labels.append('Conversion')
                leg = bax.legend(handles, labels, loc='lower left',
                                  bbox_to_anchor=(1, 0.2), labelspacing=1.0,
                                  fontsize=7, handler_map={plt.Line2D: HandlerLine2D(numpoints=3)})
                for line in leg.get_lines():
                    line.set_linewidth(1.5)

    if standalone:
        if save_path is not None:
            plt.savefig(f'{save_path}/surv_curves_{img_id}.pdf')
        plt.show()
def draw_curves_grid(full_data, full_curves_dict, n_per_patient=5,
                      n_rows=4, n_cols=4, first_conversion_idx=None,
                      marker_panel_idx=None, save_path=None, stylef=None,
                      show_legend_once=True):
    n_patients = len(full_data['image_path']) // n_per_patient
    assert n_patients <= n_rows * n_cols, \
        f"{n_patients} patients don't fit in a {n_rows}x{n_cols} grid"

    fig = plt.figure(figsize=(n_cols * n_per_patient * 2.2, n_rows * n_per_patient * 0.9))
    outer_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.5, hspace=0.7)

    with plt.style.context(stylef):
        for p in range(n_patients):
            start, end = p * n_per_patient, (p + 1) * n_per_patient

            data_to_plot = {
                'image_path': full_data['image_path'][start:end],
                'diagnosis_amd_grade': full_data['diagnosis_amd_grade'][start:end],
                'visit_number': full_data['visit_number'][start:end],
            }
            curves_dict = {
                model: curves[start:end] for model, curves in full_curves_dict.items()
            }

            row, col = divmod(p, n_cols)
            draw_curves_grid(
                data_to_plot, curves_dict,
                first_conversion_idx=first_conversion_idx,
                marker_panel_idx=marker_panel_idx,
                fig=fig, subplot_spec=outer_gs[row, col],
                show_legend=(not show_legend_once) or (p == n_patients - 1)
            )

    if save_path is not None:
        plt.savefig(f'{save_path}/surv_curves_grid.pdf', bbox_inches='tight')
    plt.show()


def plot_one_image_curve(or_df, test, freeze_encoder = 1, train_size = 500, 
                         num_images = 2, has_event = True, CHKPT_DIR = '', num0 = None):
    dfl = or_df[(or_df['freeze_encoder'] == freeze_encoder) & (or_df['survival_head'] == 'mlp') & (or_df['train_size'] == train_size)]
    dfl_ssl = dfl.groupby(['weights_path']).agg(list)

    dfl_ssl['model'] = dfl_ssl['summary'].apply(lambda x: x[0])
    model_list = sorted(list(dfl_ssl['model'].unique()))
    # print('model list',model_list)
    dfl_ssl['best_idx'] = dfl_ssl['test/ibs'].apply(lambda x: np.argmin(np.array(x)) )

    dfl_ssl['best_result'] = dfl_ssl.apply(lambda row: row['Name'][row['best_idx']], axis=1)
    dfl_ssl['best_result'].head()

    test['converter'] = test['diagnosis_amd_grade'].apply(lambda x: x in [10, 11, 12])
    test_grp = test.sort_values(['visit_number']).groupby(['eye_id', 'image_eye']).agg(list)
    test_grp['n_img'] = test_grp['image_side'].apply(lambda x: len(x))
    test_grp['n_img'].value_counts()

    test_grp['has_event'] = test_grp['event'].apply(lambda x: 1 in x)
    test_grp['has_event'].value_counts()

    # print(test_grp.columns)

    sample_ = test_grp[(test_grp['n_img'] == 5) & (test_grp['has_event'] == has_event)].reset_index(drop = False)
    # print('sample', sample_[['patient_id', 'visit_number', 'converter', 'has_event', 'diagnosis_amd_grade']])
    if num0 is None:
        num0 = random.randint(0,(len(sample_)-1))
    print('num0 is', num0)

    all_images_to_plot =  sample_.loc[num0, ['image_path', 'diagnosis_amd_grade', 'patient_id', "visit_number", "converter"]]
    # print('all_images_to_plot', all_images_to_plot)

    all_images_to_plot = pick_same_index_across_lists(all_images_to_plot)

    print('--------------------------------------')
    print()
    print(all_images_to_plot)
    print('--------------------------------------')
    print()
    curves_dict ={}

    curves_dict = get_all_curves(all_images_to_plot, model_list, dfl_ssl, CHKPT_DIR)
    return curves_dict, all_images_to_plot

def draw_curves_all_patients(full_data, full_curves_dict, id_col='ID',
                              first_conversion_idx=None, marker_panel_idx=None,
                              save_path=None, stylef=None):
    """
    full_data       : dict/df-like containing 'image_path', 'diagnosis_amd_grade', and id_col,
                       with one entry per visit across ALL individuals
    full_curves_dict: {model_name: [curve_patient1_visit0, curve_patient1_visit1, ..., curve_patient2_visit0, ...]}
                       i.e. same structure as before but spanning all individuals
    """
    ids = np.array(full_data[id_col])
    unique_ids = pd.unique(ids)

    for pid in unique_ids:
        mask = [i for i, v in enumerate(ids) if v == pid]

        # Slice out just this patient's visits
        data_to_plot = {
            'image_path': [full_data['image_path'][i] for i in mask],
            'diagnosis_amd_grade': [full_data['diagnosis_amd_grade'][i] for i in mask],
            'visit_number': [full_data['visit_number'][i] for i in mask],
        }

        curves_dict = {
            model: [curves[i] for i in mask]
            for model, curves in full_curves_dict.items()
        }

        draw_curves(
            data_to_plot,
            curves_dict,
            first_conversion_idx=first_conversion_idx,
            marker_panel_idx=marker_panel_idx,
            save_path=save_path,
            stylef=stylef
        )


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


    all_img_to_plot = data_to_plot['image_path']
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


def get_all_curves(all_images_to_plot0, model_list, dfl_ssl, CHKPT_DIR):
    print(all_images_to_plot0)
    all_images_to_plot0 = all_images_to_plot0['image_path']
    curves_dict = {}
    for img_to_plot in all_images_to_plot0:
        for m in model_list:
            surv_curv = get_model_curves(dfl_ssl, m, img_to_plot, CHKPT_DIR)
            if m not in curves_dict:
                curves_dict[m]=[]
            curves_dict[m].append(surv_curv)
    print('len of curves_dict', curves_dict)
    return curves_dict