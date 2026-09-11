from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import os

from itertools import product
import pandas as pd

def get_remnants(df, survival_head = 'mlp', freeze_encoder = 0 ):
  df = df[(df['survival_head'] == survival_head) & (df['freeze_encoder'] == freeze_encoder) ]
  expected_train_size = [500, 1000, 10000, 20000, 30000, 32500]
  expected_train_size = [float(i) for i in expected_train_size]

  models = df['summary'].unique()
  expected = pd.DataFrame(
      list(product(models, expected_train_size)),
      columns=['summary', 'train_size']
  )
  run_counts = df.groupby(['summary', 'train_size']).size().reset_index(name='run_count')

  # Merge to find missing or incomplete
  result = expected.merge(run_counts, on=['summary', 'train_size'], how='left')
  result['run_count'] = result['run_count'].fillna(0).astype(int)

  missing_or_incomplete = result[result['run_count'] < 5]
  return missing_or_incomplete

def plot_finetuning_strategies(all_plots, label2idx, 
                               cm_long, norm, metrics='test/ibs', more_suptitle = '', 
                               show_count = False, show_mlp_legend = False, save_path = '',
                                   stylef = "mpl.mplstyle"
):
    with plt.style.context(stylef):
        fig, ax = plt.subplots(1, 2, figsize = (6, 3), sharey=True)
        for a in ax:
            a.set_xscale('log')

       
        for key, value_ in all_plots.items():
            value = value_.groupby(['train_size'])[metrics].agg(['mean', 'std', 'count']).reset_index()
            value.rename(columns={'mean': metrics, 'std': 'std', 'count': 'count'}, inplace=True)

            legend_lbl = "_".join(key.split("_")[0:-1])
           
            idx   = label2idx[legend_lbl]  
            
            if "fet" in key:
                # ax_index = 0
                colour = cm_long(norm(idx))
                if "lin" in key:
                    legend_lbl_lin = legend_lbl#.replace(substring_lin, "")
                    ax[0].plot(value['train_size'], value[metrics], '.--', label=legend_lbl_lin, color = colour, linewidth=0.5) 
                    # ax[0].legend( frameon= True, ncol = 2)
                    ax[0].set_title("Freezing Encoder")
                else:     
                    legend_lbl_mlp = legend_lbl#.replace(substring_mlp, "")
                    ax[0].plot(value['train_size'], value[metrics], '.-', label=legend_lbl_mlp, color = colour)  
                    ax[0].set_title("Freezing Encoder")       
                if show_count:
                    x = value['train_size'].values
                    y = value[metrics].values
                    for x_val, y_val, count_val in zip(x, y, value['count'].values):
                        ax[0].text(x_val, y_val, str(count_val), fontsize=5)
                ax[0].legend(frameon= True, ncol = 2)
                # ax[0].set_ylim(0.045, 0.12)


            elif "fef" in key:
                # ax_index = 1
                colour = cm_long(norm(idx))
                if "lin" in key:
                    legend_lbl_lin = legend_lbl#.replace(substring_lin, "")
                
                    ax[1].plot(value['train_size'], value[metrics], '.--', label=legend_lbl_lin, color = colour,linewidth=0.5)  
                    ax[1].set_title("Not Freezing Encoder")
                else:     
                    legend_lbl_mlp = legend_lbl#.replace(substring_mlp, "")
        
                    ax[1].plot(value['train_size'], value[metrics], '.-', label=legend_lbl_mlp, color = colour)  
                    ax[1].set_title("Not Freezing Encoder")
                if show_count:
                    x = value['train_size'].values
                    y = value[metrics].values
                    for x_val, y_val, count_val in zip(x, y, value['count'].values):
                        ax[1].text(x_val, y_val, str(count_val), fontsize=5)
                # ax[1].set_ylim(0.045, 0.14)
                ax[1].legend(frameon= True, ncol = 2)

        if show_mlp_legend:   
            style_legend_lines = [
                Line2D([0], [0], color='black', linestyle='-', label='MLP Head',linewidth=1.0),
                Line2D([0], [0], color='black', linestyle='--', label='Linear Head',linewidth=1.0)
            ]

            # Place style legend in empty space or below
            fig.legend(handles=style_legend_lines,
                    loc='lower center',
                    ncol=1,
                    frameon=True,
                    bbox_to_anchor=(0.1, 0.8))
        metric_val = metrics.split("/")[1].upper() if "/" in metrics else metrics.upper()
        fig.supxlabel('Train Size', fontsize=5)
        fig.supylabel(f'Test {metric_val}', fontsize=5)
        if len(more_suptitle) > 0:
            plt.suptitle(f'{more_suptitle}', fontdict={"fontsize": 6})
        else:
            plt.suptitle('Finetuning Strategies', fontdict={"fontsize": 6})
        if len(save_path) > 1:
            plt.savefig(f'{save_path}', dpi = 300)
        # plt.axis('equal')
        plt.show()


def display_longitudinal_data(train, image_folder,
                               identifier = 'eye_id', 
                                image_path = 'image_path', 
                                disease_name = 'diagnosis_amd_grade',
                                visit_number = 'visit_number', 
                                image_side = 'image_side',
                                image_field = 'image_field',
                                prefix='', n_samples = 5, n_images_per_id = 5,
                                suptitle = 'Longitudinal Data Visualization'):
    images_to_view = {}
    random_ids = random.sample(list(range(train.shape[0])), n_samples)
    for i in random_ids:
        row = train.iloc[i]
        id = row[identifier]
        image_list = row[image_path]
        # print('image_list', image_list)
        len_images = min(10, len(image_list))
        all_labels = row[disease_name]
        session_vals = row[visit_number]
        image_sides = row[image_side]
        image_fields = row[image_field]
        for j in range(len_images):
            image_name = image_list[j]
            label = all_labels[j]
            session_val = session_vals[j]
            img_side = image_sides[j]
            img_field = image_fields[j]
            image  = Image.open(os.path.join( image_folder, f"{prefix}{image_name}")).convert("RGB")
            # print('image size',image.size)
            if id not in images_to_view:
                images_to_view[id] = {'images': [], 'label': [], 'img_side': [], 'session_val': [], 'img_field': [] }
            if len(images_to_view[id]['images']) < n_images_per_id:
                images_to_view[id]['images'].append(image)
                images_to_view[id]['session_val'].append(session_val)
                images_to_view[id]['label'].append(label)
                images_to_view[id]['img_side'].append(img_side)
                images_to_view[id]['img_field'].append(img_field)
    plt_width = max(len(images_to_view[id]['images']) for id in images_to_view) + 1
    print('plt_width', plt_width)
    fig, axs = plt.subplots(n_samples, plt_width, figsize=(plt_width * 2, 5))
    for t, (id, data) in enumerate(images_to_view.items()):
        axs[t, 0].text(0,0.3, id)
        axs[t, 0].axis('off')
        n_images_in_id = len(data['images'])        
        for k in range(n_images_in_id): 
            image = data['images'][k] 
            amount_ = data['session_val'][k]
            label = data['label'][k]
            image_side_ = data['img_side'][k]
            img_field_ = data['img_field'][k]
            image_arr = np.array(image)
            title = f"{image_side_} DR: {label}" #F: {img_field_}
            axs[t, k+1].imshow(image_arr)
            axs[t, k+1].set_title(title, fontsize = 6, pad = 2)
            axs[t, k+1].axis('off')
    
    for a in axs.flatten():
        a.axis('off')
    plt.suptitle(f'{suptitle}', fontdict={"fontsize": 8})
    plt.show()

def display_areds(train, image_folder, prefix='', n_samples = 5, n_images_per_id = 5):
    images_to_view = {}
    random_ids = random.sample(list(range(train.shape[0])), n_samples)
    print(random_ids)
    for i in random_ids:
        row = train.iloc[i]
        id = row['eye_id']
        image_list = row['image_path']
        len_images = min(10, len(image_list))
        for j in range(len_images):
            image_name = image_list[j]
            label = row['diagnosis_amd_grade'][j]
            image_eye = row['image_eye'][j]
            visit_number = row['visit_number'][j]
            image_side = row['image_side'][j]
            image  = Image.open(os.path.join( image_folder, f"{prefix}{image_name}")).convert("RGB")

            if id not in images_to_view:
                images_to_view[id] = {'images': [], 'visit_number': [], 'image_eye': [], 'label': [], 'image_side': []}
            if len(images_to_view[id]['images']) < n_images_per_id:
                images_to_view[id]['images'].append(image)
                images_to_view[id]['visit_number'].append(visit_number)
                images_to_view[id]['image_eye'].append(image_eye)
                images_to_view[id]['label'].append(label)
                images_to_view[id]['image_side'].append(image_side)

    fig, axs = plt.subplots( n_samples, n_images_per_id+1, figsize=(n_images_per_id * 2, n_samples))
    for t, (id, data) in enumerate(images_to_view.items()):
        axs[t, 0].text(0,0.3, id)
        axs[t, 0].axis('off')
        n_images_in_id = len(data['images'])
        
        for k in range(n_images_in_id): 
            
            image = data['images'][k] 
            image_eye = data['image_eye'][k]
            amount_ = data['visit_number'][k]
            label = data["label"][k]
            image_side = data["image_side"][k]

            image_arr = np.array(image)
            print('image shape is', image_arr.shape)
            
            title = f"{image_eye} {image_side} V_no: {amount_} AMD: {label}"
            axs[t, k+1].imshow(image_arr)
            axs[t, k+1].set_title(title, fontsize = 6)
        
            axs[t, k+1].axis('off')