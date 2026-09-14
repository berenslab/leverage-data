import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brokenaxes import brokenaxes
import seaborn as sns
import string
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from all_utils import get_color_legend, MODEL_GROUPS, NEW_NAMES, \
    make_publication_legend_columns


CURVE_KEY_ALIASES = {
    'scratch':          'resnet_scratch',
    'resnet_imagenet':  'resnet_INet',
    'simclr_inet_nako': 'simclr_INet_nako',
}

def normalize_model_key(key):
    return CURVE_KEY_ALIASES.get(key, key)

def get_patient_panel(or_df, test, row_idx, freeze_encoder, train_size, has_event, CHKPT_DIR,
                      patient_id, survival_head):
    dict1, img_to_plot, train_size_returned = plot_one_image_curve(
        or_df, test, freeze_encoder=freeze_encoder, train_size=train_size,
        num_images=None, has_event=has_event, CHKPT_DIR=CHKPT_DIR,
        to_plot_row_index=row_idx, patient_id=patient_id, survival_head=survival_head
    )

    grades = img_to_plot['diagnosis_amd_grade']
    visit_number = [int(v / 2) for v in img_to_plot['visit_number']]

    marker_x = None
    is_conversion = False

    if any(g in [10, 11, 12] for g in grades):
        first_conversion_idx = next((i for i, g in enumerate(grades) if g >= 10), None)
        panel_idx = max(first_conversion_idx - 2, 0)          # pre-conversion visit
        marker_x = visit_number[first_conversion_idx] - visit_number[panel_idx]
        is_conversion = True
    else:
        panel_idx = min(2, len(grades) - 1)                    # fixed reference visit

    model_curves = {model: curves[panel_idx] for model, curves in dict1.items()}
    return {
        'patient_id': img_to_plot['patient_id'][panel_idx],
        'image_path': img_to_plot['image_path'][panel_idx],
        'conversion_img': img_to_plot['image_path'][-1],
        'diagnosis_amd_grade': grades[panel_idx],
        'visit_number': visit_number[panel_idx],
        'marker_x': marker_x,
        'is_conversion': is_conversion,
        'train_size': train_size_returned,
        'curves': model_curves,
    }
    


def pick_same_index_across_lists(row, num_images=2):
    list_lengths = [len(v) for v in row if isinstance(v, list)]
    if not list_lengths:
        return row

    n = list_lengths[0]

    if num_images is None:
        chosen_idxs = list(range(n))  # everything
    else:
        chosen_idxs = list(range(max(0, n - num_images), n))  # last num_images indices

    new_row = row.copy()
    for k, v in row.items():
        if isinstance(v, list):
            new_row[k] = [v[i] for i in chosen_idxs]
    return new_row



def get_model_curves(df, model_name, img_to_plot, CHKPT_DIR):

    row = df[(df['model']==model_name)]
    exp_best = row['best_result'].values[0]
    full_path = os.path.join(CHKPT_DIR, exp_best)
    exp_df = pd.read_csv(os.path.join(full_path, 'test_survival_curves.csv' ))
    image_curve = img_to_plot
    image_curve = image_curve.split('/')[-1]
    h = exp_df[exp_df['image_name'] == image_curve ]
    s = h['survival_curve'].values[0].strip().strip('[]')
    surv_curv = [float(x) for x in s.split()]
    return surv_curv


def get_all_curves(all_images_to_plot0, model_list, dfl_ssl, CHKPT_DIR):
    all_images_to_plot0 = all_images_to_plot0['image_path']
    curves_dict = {}
    for img_to_plot in all_images_to_plot0:
        for m in model_list:
            surv_curv = get_model_curves(dfl_ssl, m, img_to_plot, CHKPT_DIR)
            if m not in curves_dict:
                curves_dict[m]=[]
            curves_dict[m].append(surv_curv)
    return curves_dict

def draw_patient_grid(records, image_dir=None, n_cols=4,
                       save_path=None, stylef=None, save_name=None,
                       legend_bbox_to_anchor=(0.5, -0.05),
                       text_x = -0.15, text_y = 1.08, show_title = True,
                        survival_head = None,
                        pos_dict = {'pos1_x0' : 0.14,
                                    'pos1_y0' : 0.095,
                                    'pos1_width' : 0.075,
                                    'pos1_height' : 0.065,
                                    'pos2_x0' : 0.13,
                                    'pos2_y0' : 0.15,
                                    'pos2_width' : 0.055,
                                    'pos2_height' : 0.065,
                        }

                ):
    assert survival_head is not None, 'please include a survival head'

    group_colors, cm_colors, color_groups = get_color_legend()
    head_styles = {
        "mlp": "-",
        "linear": "--",
    }
    
    y_min = min(min(curve) for r in records for curve in r['curves'].values()) - 0.02
    y_max = 1
    x_max = max(len(next(iter(r['curves'].values()))) for r in records)


    present_models = set()
    for all_plots in records:
        for key in all_plots['curves'].keys():
            present_models.add((normalize_model_key(key)))


    print(set(color_groups) - {m for r in records for m in r['curves']})
    print({m for r in records for m in r['curves']} - set(color_groups))
    show_images = image_dir is not None
    n_rows_layout = 2 if show_images else 1
    height_ratios = [3, 1] if show_images else [1]

    style_ctx = stylef if stylef is not None else 'default'
    
    width_pt = 452.9679
    width_in = width_pt / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    height_in = (width_in / 2) * golden_ratio * 2 * 1.3
    with plt.style.context(style_ctx):
        fig = plt.figure(figsize=(width_in, 3.5))

        gs = gridspec.GridSpec(
                    n_rows_layout, n_cols, figure=fig,
                    wspace=0.35, hspace=0.35,
                    height_ratios=height_ratios,
                    left=0.08, right=0.98,      

        )

        for p, rec in enumerate(records):
            bax = brokenaxes(
                xlims=((0.05, 0.002), (0.9, x_max + 0.1)),
                width_ratios=[0.05, 1],
                hspace=0.001, d=0.003, tilt=45,
                subplot_spec=gs[0, p], fig=fig
            )
            
            for line in bax.diag_handles:
                line.set_linewidth(1.)

            panel_letter = string.ascii_uppercase[p]
            bax.axs[0].text(text_x, text_y, panel_letter, 
                             fontsize=10, fontweight='bold', 
                             va='top',
                               ha='left', clip_on=False,
                               fontname = 'Arial')

            for model, curve in rec['curves'].items():
                canon = normalize_model_key(model)
                bax.plot(range(1, len(curve) + 1 ), curve,
                          label=NEW_NAMES.get(canon, canon),
                          color=color_groups.get(canon), linewidth=1.5,
                          linestyle = head_styles[survival_head])

            if rec.get('is_conversion') and rec.get('marker_x') is not None:
                bax.axs[1].plot(rec['marker_x'], 0.02, marker='v', markersize=6,
                                 color='grey', linestyle='none', clip_on=False, zorder=5)
          
            bax.set_ylim(y_min, y_max)
            

            bax.axs[0].set_xticks([])
            bax.axs[1].set_xticks(range(1, x_max + 1))
            for ax in bax.axs:
                ax.tick_params(axis='both', labelsize=10, width=1, length=2)
                ax.set_yticks([0, 0.5, 1.0])
                for spine in ['left', 'bottom']:
                    ax.spines[spine].set_linewidth(1)

          
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
            from PIL import Image
            # size = ((28, 28))


            if p == 0:
                inner_gs = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=gs[1, p], wspace=10.5
                )
                ax_img1 = fig.add_subplot(inner_gs[0, 0])
                ax_img2 = fig.add_subplot(inner_gs[1, 0])

                pos = ax_img1.get_position()
                
                ax_img1.set_position([pos.x0-pos_dict['pos1_x0'], 
                                      pos.y0 - pos_dict['pos1_y0'],
                                       pos.width + pos_dict['pos1_width'], 
                                       pos.height + pos_dict['pos1_height']])

                try:
                    img_path1 = os.path.join(image_dir, f"R_{rec['image_path']}")
                    img_path1 = Image.open(img_path1)#.resize((224, 224))
                    ax_img1.imshow(img_path1)
                except Exception as e:
                    print(f"Could not load image {rec['image_path']}: {e}")
                ax_img1.axis('off')

                try:
                    pos1 = ax_img2.get_position()
                    ax_img2.set_position([pos1.x0 - pos_dict['pos2_x0'], pos1.y0 - pos_dict['pos2_y0'],
                                           pos1.width + pos_dict['pos2_width'], pos1.height +pos_dict['pos2_height']])
                    img_path2 = os.path.join(image_dir, f"R_{rec['conversion_img']}")
                    img_path2 = Image.open(img_path2)
                    ax_img2.imshow(img_path2)
                except Exception as e:
                    print(f"Could not load image {rec['conversion_img']}: {e}")
                ax_img2.axis('off')

                

            if p == 0:
                fig.text(0.01, 0.64, 'S(t)', fontsize=10, rotation='vertical',
              va='center', ha='left')
           
        group_columns = make_publication_legend_columns(cm_colors, MODEL_GROUPS,
                                                         group_colors, present_models )

        fm_in_handles,  fm_in_labels,  fm_in_raw  = group_columns[0]
        fm_out_handles, fm_out_labels, fm_out_raw = group_columns[1]

        merged_fm_column = (
            fm_in_handles + fm_out_handles,
            fm_in_labels  + fm_out_labels,
            fm_in_raw     + fm_out_raw,
        )

        group_columns = [merged_fm_column] + group_columns[2:]   

        bs_handles, bs_labels, bs_raw = group_columns[-1]
        bs_handles = bs_handles + [Line2D([0, 1, 2], [0, 0, 0], marker='v', color='grey',
                                            markerfacecolor='grey', markersize=6, linestyle='none')]
        bs_labels = bs_labels + ['Conversion']
        bs_raw = bs_raw + [None]
        group_columns[-1] = (bs_handles, bs_labels, bs_raw)   

        all_columns = group_columns   
        max_rows = max(len(h) for h, l, r in all_columns)


        flat_handles, flat_labels, flat_raw = [], [], []
        for handles, labels, raw in all_columns:
            pad = max_rows - len(handles)
            handles = handles + [Line2D([0], [0], color='none')] * pad
            labels  = labels  + [''] * pad
            raw     = raw     + [None] * pad
            flat_handles.extend(handles)
            flat_labels.extend(labels)
            flat_raw.extend(raw)

            
        combined_legend = fig.legend(
            handles=flat_handles,
            labels=flat_labels,
            loc='upper center',
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=len(all_columns),
            prop={'family': 'Arial', 'size': 10},   
            edgecolor='#cccccc',
            frameon=False,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.5,
            handler_map={Line2D: HandlerLine2D(numpoints=3)},
        )

        for text, raw_label in zip(combined_legend.get_texts(), flat_raw):
            if raw_label is not None:
                text.set_fontweight('bold')

    fig.supxlabel('Years', fontsize=10, x=0.5, y=0.285, family = 'Arial')
   
    if save_path is not None:
        fname = save_name or 'patient_grid'
        plt.savefig(f'{save_path}/{fname}.pdf', bbox_inches='tight', dpi =300)
    plt.show()



def plot_one_image_curve(or_df, test, freeze_encoder = 1, 
                         train_size = 30000, 
                         num_images = None, has_event = True, CHKPT_DIR = '', 
                         to_plot_row_index = None, patient_id = None,
                         return_only_pid = False,
                         n_img_values=(3, 4, 5, 6), survival_head = None):

    assert survival_head is not None, 'enter the survival head'
    dfl = or_df[(or_df['freeze_encoder'] == freeze_encoder) & (or_df['survival_head'] == survival_head) & (or_df['train_size'] == train_size)]
    dfl_ssl = dfl.groupby(['weights_path']).agg(list)

    dfl_ssl['model'] = dfl_ssl['summary'].apply(lambda x: x[0])
    model_list = sorted(list(dfl_ssl['model'].unique()))

    dfl_ssl['best_idx'] = dfl_ssl['test/ibs'].apply(lambda x: np.argmin(np.array(x)) )

    dfl_ssl['best_result'] = dfl_ssl.apply(lambda row: row['Name'][row['best_idx']], axis=1)

    test['converter'] = test['diagnosis_amd_grade'].apply(lambda x: x in [10, 11, 12])
    test_grp = test.sort_values(['visit_number']).groupby(['eye_id', 'image_eye']).agg(list)
    test_grp['n_img'] = test_grp['image_side'].apply(lambda x: len(x))

    test_grp['has_event'] = test_grp['event'].apply(lambda x: 1 in x)

    sample_ = test_grp[(test_grp['n_img'].isin(n_img_values)) & (test_grp['has_event'] == has_event)].reset_index(drop = False)
    
    if return_only_pid:
        return list(sample_['patient_id'])
    if patient_id is not None:
        matches = sample_.index[sample_['patient_id'].apply(lambda x: x[0] if isinstance(x, list) else x) == patient_id].tolist()
        if not matches:
            raise ValueError(f"patient_id {patient_id} not found in filtered sample_ (freeze_encoder={freeze_encoder}, train_size={train_size}, has_event={has_event})")
        to_plot_row_index = matches[0]
    elif to_plot_row_index is None:
        to_plot_row_index = random.randint(0, (len(sample_) - 1))

    
    all_images_to_plot =  sample_.loc[to_plot_row_index, ['image_path', 'diagnosis_amd_grade', 'patient_id', "visit_number", "converter"]]
    all_images_to_plot['train_size'] = train_size

    all_images_to_plot = pick_same_index_across_lists(all_images_to_plot, num_images = num_images)
    
    curves_dict ={}

    curves_dict = get_all_curves(all_images_to_plot, model_list, dfl_ssl, CHKPT_DIR)
    print('curves_dict', curves_dict)
    return curves_dict, all_images_to_plot, train_size