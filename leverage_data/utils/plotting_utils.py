

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
import json
import os
import telegram
from pathlib import Path
from dotenv import load_dotenv
import os
from PIL.PngImagePlugin import PngImageFile

load_dotenv()

telegram_pin = os.environ.get('TELEGRAM_PIN')
def get_token_chat_id():
    with open(telegram_pin, 'r') as file:
        config = json.load(file)
    return config


class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        # self.rc = get_token_chat_id()

        self.token = os.environ.get('telegram_token')
        self.chat_id = os.environ.get('telegram_chat_id')
        self.bot = telegram.Bot(token=self.token)

    async def send(self, fname):
        fname = Path(fname)
        ext = fname.suffix[1:]
        
        if ext == "png":
            return await self.send_png(fname)
        elif ext == "txt" or ext == "md":
            return await self.send_txt(fname)
        else:
            raise RuntimeError(
                "File ext '{}' not understood, file {} not sent".format(
                    ext, fname
                )
            )

    async def send_png(self, fname):
        png = PngImageFile(fname)

        png.load()  # load metadata
        with open(fname, "rb") as f:
            return await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=f,
                parse_mode="Markdown",
                caption=png.info.get("Comment", "`{}`".format(fname)),
            )

    async def send_txt(self, fname):
        filename = f"`{fname}`\n`---`\n"
        filecontent = fname.read_text()

        text = filename + filecontent
        return await self.bot.send_message(
            chat_id=self.chat_id, text=text, parse_mode="Markdown"
        )

def create_legend(unique_label, cm, ax, legend_title, location = 'upper left', markersize = 5, n_col = 2, fontsize = 7):
    handles = []
    for i, label in enumerate(unique_label):
        handles.append(plt.Line2D([], [], marker='o', color=cm(i / len(unique_label)), linestyle='None', markersize=markersize, label=label))
    return ax.legend(handles=handles,loc = location,ncol=n_col, frameon = False,title=legend_title, fontsize = fontsize)
      
async def plot_embeddings(results, 
                        embeddings, 
                        labels, 
                        plot_title, 
                        experiment_directory,
                        stylef =None):
    print('plotting embeddings')
    
    # acc1, auc1 = linear_acc(X = embeddings, 
    #                       y = labels,
    #                       id = id,
    #                       target_label_name = "label",
    #                       SPLIT_SEED = 10, 
    #                     evaluate_weights=False ) 
    
    acc0 = results['acc0']
    auc0 = results['auc0']

    acc1 = results['acc1']
    auc1 = results['auc1']

    print('unique labels are', np.unique(labels))
    n_unique_labels = len(np.unique(labels))
    if n_unique_labels == 2:
        label_colors = {
            0: "red",
            1: "blue",
        }
        unique_labels = sorted(label_colors.keys())       # [0, 1]
        colors = [label_colors[k] for k in unique_labels] # ["red", "blue"]
        cm = ListedColormap(colors)
        with plt.style.context(stylef): 
            fig, ax = plt.subplots(figsize=(4, 4))
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=cm, alpha=0.8)  
            create_legend(unique_label = unique_labels, cm = cm, ax = ax, legend_title = '')

            f_plot_title = f'{plot_title} H_acc. = {acc0:.1f}% H_auc. = {auc0:.1f} acc. = {acc1:.1f}% auc. = {auc1:.1f}'
            plt.title(f_plot_title)
            
            s_path = f"{experiment_directory}/{acc1:.1f}_{auc1:.1f}"
            plt.savefig(f'{s_path}.png')

        message = TelegramBot()
        await message.send(f'{s_path}.png')

    else:
        cm = plt.get_cmap('tab20c')

        with plt.style.context(stylef): 
            fig, ax = plt.subplots(figsize=(4, 4))
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=cm, alpha=0.8)  
    
            f_plot_title = f'{plot_title} acc. = {acc1:.1f}% auc. = {auc1:.1f}'
            plt.title(f_plot_title)
            
            s_path = f"{experiment_directory}/{acc1:.1f}_{auc1:.1f}"
            plt.savefig(f'{s_path}.png')

        message = TelegramBot()
        await message.send(f'{s_path}.png')


# def draw_curves(all_img_to_plot, curves_dict, first_conversion_idx = None,
#                  marker_panel_idx = None, save_path = None):
#     """
#     all_img_to_plot : [images, labels, [img_id]]
#                       labels = list of diagnosis_amd_grade per visit
#     curves_dict     : {model_name: [curve_visit_0, curve_visit_1, ...]}

#     Conversion marker: a single grey downward triangle on the x-axis at the
#     conversion year, shown only in the penultimate panel (last pre-conversion visit).
#     """
#     color_groups = {    
#     'dino_nako': '#000000',
#     'dino_meta': '#05771D',
#     'retfound':  '#90A4AE',  # fixed typo
#     'nako_mae':  '#6A1B9A',
#     'simclr_inet_nako': '#D81B60',
#     'simclr_nako': '#F3D0DC',
#     'scratch': '#1565C0',
#     'resnet_imagenet': '#42A5F5',
#                     }
#     new_names = {
#         'dino_nako': 'DINOv2 NAKO',
#         'dino_meta': 'DINOv2 LVD',
#         'retfound': 'RETFound',
#         'resnet_imagenet': 'ResNet INet',
#         'nako_mae': "MAE NAKO",
#         'simclr_nako': 'SimCLR NAKO',
#         'simclr_inet_nako': 'SimCLR INet NAKO',
#         'scratch': "ResNet Scratch"
        
#     }
    
#     stylef = '/home/berens/bep973/ifeoma_home/survival_modeling/survival_on_embedding/survival_on_embedding/utils/berenslab.mplstyle'

#     fig = plt.figure(figsize=(len(all_img_to_plot[0])*3, 4))
#     gs = gridspec.GridSpec(1, len(all_img_to_plot[0]), figure=fig, wspace=0.3)

#     y_min = min(min(curve) for curves in curves_dict.values() for curve in curves) - 0.02
#     y_max = 1
#     x = [1, 2, 3, 4, 5]

#     # ── Derive conversion marker position ─────────────────────────────────────
#     grades = all_img_to_plot[1]   # e.g. [7.0, 8.0, 9.0, 9.0, 11.0]
#     visit_number = all_img_to_plot[-1]

#     if any(g in [10, 11, 12] for g in grades):

#         if first_conversion_idx is None:
#             first_conversion_idx = next(
#                 (i for i, g in enumerate(grades) if g >= 10), None
#             )

#             if first_conversion_idx is not None and first_conversion_idx > 0:
#                 marker_panel_idx = first_conversion_idx - 1   # penultimate panel
#                 marker_x         = first_conversion_idx + 1   # 1-based conversion year
#             else:
#                 marker_panel_idx = None
#                 marker_x         = None
#         else:
#             marker_x = first_conversion_idx
#             # marker_panel_idx = first_conversion_idx  -2
#         print(f'marker x is {marker_x} marker_panel_idx is {marker_panel_idx}')
#     # ──────────────────────────────────────────────────────────────────────────

#     with plt.style.context(stylef):
#         for i, img in enumerate(all_img_to_plot[0]):
#             bax = brokenaxes(
#                 xlims=((-0.1, 0.002), (0.9, max(x) + 0.1)),
#                 hspace=0.01,
#                 d=0.001,
#                 tilt=45,
#                 subplot_spec=gs[i],
#                 fig=fig
#             )

#             for model, curves in curves_dict.items():
#                 curve_to_plot = curves[i]
#                 bax.plot(
#                     range(1, len(curve_to_plot) + 1),
#                     curve_to_plot,
#                     label=new_names[model],
#                     color=color_groups[model],
#                     linewidth=1.5
#                 )

#             # ── Single conversion marker on the x-axis ────────────────────────
#             if i == marker_panel_idx:
#                 # bax.axs[1] is the main (right) axes where the curves are drawn
#                 bax.axs[1].plot(
#                     marker_x, 0.02,              # sit right on the x-axis
#                     marker='v',
#                     markersize=10,
#                     color='grey',
#                     linestyle='none',
#                     clip_on=False,            # don't clip if right at the edge
#                     zorder=5
#                 )
#             # ──────────────────────────────────────────────────────────────────

#             img_id = all_img_to_plot[2][0]
#             bax.set_title(f"ID {img_id} V_no {visit_number[i]} Label {int(grades[i])}", fontsize=10)
#             bax.set_ylim(y_min, y_max)
#             if i == 0:
#                 bax.set_ylabel('S(t)', fontsize=10)       # y-label on the leftmost panel only
#             if i == len(all_img_to_plot[0]) // 2:
#                 bax.set_xlabel('Year', fontsize=10) 
#             bax.axs[0].set_xticks([])
#             bax.axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
#             for ax in bax.axs:
#                 ax.tick_params(axis='both', labelsize=10)
#                 ax.set_yticks([0, 0.5, 1.0])

#             if i == len(all_img_to_plot[0]) - 1:
#                 handles, labels = [], []
#                 for model in curves_dict:
#                     handles.append(plt.Line2D([0], [0],
#                                               color=color_groups[model],
#                                               linewidth=1.5))
#                     labels.append(new_names[model])

#                 if marker_panel_idx is not None:
#                     handles.append(plt.Line2D([0], [0],
#                                               marker='v', color='w',
#                                               markerfacecolor='grey',
#                                               markersize=8,
#                                               linestyle='none'))
#                     labels.append('Conversion')

#                 leg = bax.legend(
#                     handles, labels,
#                     loc='lower left',
#                     bbox_to_anchor=(1, 0.2),
#                     labelspacing=1.5,
#                     fontsize=8,
#                     handler_map={plt.Line2D: HandlerLine2D(numpoints=3)}
#                 )
#                 for line in leg.get_lines():
#                     line.set_linewidth(1.5)
#     if save_path is not None:
#         plt.savefig(f'{save_path}/surv_curves_{img_id}.pdf')
#     plt.show()

