

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
import matplotlib as mpl
plt.rcParams.get('font.family', 'arial')
# mpl.rc_file('../../.matplotlibrc')

from .plotting_helpers import get_axis_limits, get_base_key, \
    get_yticks, add_axis_break, make_publication_legend, make_publication_legend_columns
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math

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
        axis_br_x_pos = -0.005,
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
        fontsize=6
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

     
        # fig, ax = plt.subplots(1, 4, figsize=(14, 3.8), layout='tight')
        width_pt = 452.9679
        width_in = width_pt / 72.27
        golden_ratio = (5**0.5 - 1) / 2
        height_in = (width_in / 2) * golden_ratio * 2 * 1.3
        fig, ax = plt.subplots(1, 4, figsize=(width_in, 1.5), layout='tight')

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
                linewidth = 1.
                is_fet     = to_plot in key          # True → solid (fet), False → dashed (lin)
                target_ax  = ax[m * 2 + (1 if is_fet else 0)]   # lin now goes to ax[m*2], fet to ax[m*2+1]
                # is_fet     = to_plot in key          # True → solid (fet), False → dashed (lin)
                # target_ax  = ax[m * 2 + (0 if is_fet else 1)]

                target_ax.plot(
                    value['train_size'], value[metrics],
                    linestyle,
                    color=colour, linewidth=linewidth, markersize=3,
                    markevery=list(range(1, len(value['train_size'])))
                )

                if show_count:
                    for x_val, y_val, c_val in zip(
                            value['train_size'], value[metrics], value['count']):
                        target_ax.text(x_val, y_val, str(c_val), fontsize=fontsize)

            print('metrics', metrics)
            if metrics == 'test/ibs':
                y_min0, y_max0, _ = get_axis_limits(all_plots, metrics)


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
                # if a is not ax[m * 2]:  # hide y ticks on right axis
                #     a.tick_params(labelleft=False, left = False)
                if a is ax[m * 2]:
                    metric_label = metrics.split('/')[1].upper() if '/' in metrics else metrics.upper()
                    metric_label = metric_label.lower()

                    a.set_ylabel(f'{metric_label_dict[metric_label]}', fontsize=fontsize, 
                                x=0.045) 
                if metrics == 'test/ibs':
                    a.set_yticks(yticks0)
                    a.set_ylim(yticks0[0], yticks0[-1])
                    # a.set_ylim(yticks0[0], 0.1)     

                    
                else:
                    a.set_yticks(yticks1)
                    # a.set_ylim(yticks1[0], yticks1[-1])
                    a.set_ylim(yticks1[0], 1.0)

                a.set_xticks([], minor=True)
                a.set_xticks([1e2, 1e3, 1e4, 1e5])
                a.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'] 
                                  )
                # a.tick_params(axis='y', labelsize=10)   
                # a.tick_params(axis='x', labelsize=10)  
                a.tick_params(axis='both', labelsize=4, 
                              width = tick_width, length = tick_length) 

                a.set_xlim(100, 1.2e5)
                sns.despine(ax=a, offset={'left': 3, 'bottom': 3}, trim=False)
                for spine in ['left', 'bottom']:   # only the spines sns.despine kept
                    a.spines[spine].set_linewidth(spine_linewidth)

                # add_axis_break(
                #     a,
                #     size=axis_br_size,
                #     gap=axis_br_gap,
                #     y_pos=axis_br_y_pos,
                #     x_pos = axis_br_x_pos 

                # )
                
        print(ax[0].yaxis.label.get_fontsize())   # should print 10 if your set_ylabel call took effect
        print(fig._supxlabel.get_fontsize() if fig._supxlabel else "no supxlabel yet")
        proxy_mlp = Line2D([0], [0], linestyle='-',  color='gray', linewidth=2, label='MLP head')
        proxy_lin = Line2D([0], [0], linestyle='--', color='gray', linewidth=2, label='Linear head')    

        
        group_columns = make_publication_legend_columns(cm_colors, model_groups, group_colors)

        head_column = (
            [Line2D([0], [0], color='none'), proxy_mlp, proxy_lin],
            ['Head type', 'MLP head', 'Linear head'],
            ['Head type', None, None],
        )

        all_columns = [head_column] + group_columns
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
            bbox_to_anchor=(0.36, -0.01),   # push below panels; tune the y-value
            ncol=len(all_columns),          # one column per group + head-type
            fontsize=fontsize,
            edgecolor='#cccccc',
            frameon=False,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        for text, raw_label in zip(combined_legend.get_texts(), flat_raw):
            if raw_label is not None:
                text.set_fontweight('bold')

        for a, label in zip(ax.flat, 'ABCDEF'):
            a.text(text_x, text_y, f'{label}', transform=a.transAxes,
            fontsize=fontsize, fontweight='bold',
            ha='left', va='top', clip_on=False)

                # <-- pull ylabel closer to axes
        fig.supxlabel('Longitudinal training set size (#images)', fontsize=fontsize, 
                                       x=0.38, y=0.01)
        fig.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(bottom=0.15)

        extra_gap = 0.01   # fraction of figure width; increase/decrease to taste
        for i, a in enumerate(ax):
            pos = a.get_position()
            shift = extra_gap if i >= 2 else 0   # shift everything from ax[2] onward to the right
            a.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
                
        plt.show()
        if save_path:
            fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        
        return fig, ax

def plot_finetuning_strategies_2x2(
        all_plots,
        all_metrics=('test/ibs', 'test/concordance_index'),
        show_count=False,
        save_path='',
        to_plot='fet',
        num_y_ticks=6,
        axis_br_size=0.012,
        axis_br_gap=0.018,
        axis_br_y_pos=-0.03,
        axis_br_x_pos=-0.02,
        y_ticks_list1=None,
        y_ticks_list2=None,
        define_ticks=True,
        text_x=-0.15,
        text_y=1.0,
        dataset='areds',
        stylef='c.mplstyle',
        spine_linewidth=1.0,
        tick_fontsize=12,
        label_fontsize=12,
        add_y_ticks = True
):
    metric_label_dict = {
        'ibs': 'Integrated Brier Score',
        'concordance_index': "Concordance Index",
    }
    with plt.style.context(stylef):

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

        # ── figure: 2x2 grid ──────────────────────────────────────────────
        # row 0 -> metric 0 (e.g. IBS),           col 0 = fet, col 1 = lin
        # row 1 -> metric 1 (e.g. concordance),   col 0 = fet, col 1 = lin
        width_pt = 452.9679
        width_in = width_pt / 72.27
        golden_ratio = (5**0.5 - 1) / 2
        height_in = (width_in / 2) * golden_ratio * 2 * 1.3
        fig, ax = plt.subplots(2, 2, figsize=(width_in, height_in), layout='tight')

        for a in ax.flat:
            a.set_xscale('log')

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
                is_fet    = to_plot in key
                target_ax = ax[m, 1 if is_fet else 0]   # fet -> col 1, lin -> col 0 (swap if you want opposite)

                target_ax.plot(
                    value['train_size'], value[metrics],
                    linestyle,
                    color=colour, linewidth=linewidth, markersize=4,
                )

                if show_count:
                    for x_val, y_val, c_val in zip(
                            value['train_size'], value[metrics], value['count']):
                        target_ax.text(x_val, y_val, str(c_val), fontsize=10)

            if metrics == 'test/ibs':
                y_min0, y_max0, _ = get_axis_limits(all_plots, metrics)
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

            for col, a in enumerate([ax[m, 0], ax[m, 1]]):
                # if col == 1:  # hide y ticks on right column
                #     a.tick_params(labelleft=False, left=add_y_ticks)
                # else:
                metric_label = metrics.split('/')[1].upper() if '/' in metrics else metrics.upper()
                metric_label = metric_label.lower()
                a.set_ylabel(f'{metric_label_dict[metric_label]}', fontsize=label_fontsize, x=0.045)

                if metrics == 'test/ibs':
                    a.set_yticks(yticks0)
                    a.set_ylim(yticks0[0], yticks0[-1])
                else:
                    a.set_yticks(yticks1)
                    a.set_ylim(yticks1[0], 1.0)
                
                a.set_xticks([], minor=True)
                a.set_xticks([1e2, 1e3, 1e4, 1e5])
                a.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
                a.tick_params(axis='both', labelsize=tick_fontsize)
                a.set_xlim(80, 1.2e5)

                sns.despine(ax=a, offset={'left': 4, 'bottom': 5}, trim=False)
                for spine in ['left', 'bottom']:
                    a.spines[spine].set_linewidth(spine_linewidth)
                a.tick_params(axis='both', width=spine_linewidth, length=5)

                # add_axis_break(
                #     a,
                #     size=axis_br_size,
                #     gap=axis_br_gap,
                #     y_pos=axis_br_y_pos,
                #     x_pos=axis_br_x_pos,
                # )

        # shared x-label, once, below the whole grid
        fig.supxlabel('Longitudinal training set size (#images)', fontsize=label_fontsize, x=0.53, y=0.02)

        # ── legend ─────────────────────────────────────────────────────────
        proxy_mlp = Line2D([0], [0], linestyle='-',  color='gray', linewidth=2, label='MLP head')
        proxy_lin = Line2D([0], [0], linestyle='--', color='gray', linewidth=2, label='Linear head')

        group_columns = make_publication_legend_columns(cm_colors, model_groups, group_colors)

        # head_column = (
        #     [Line2D([0], [0], color='none'), proxy_mlp, proxy_lin],
        #     ['Head type', 'MLP head', 'Linear head'],
        #     ['Head type', None, None],
        # )

        # all_columns = [head_column] + group_columns
        # max_rows = max(len(h) for h, l, r in all_columns)

        # flat_handles, flat_labels, flat_raw = [], [], []
        # for handles, labels, raw in all_columns:
        #     pad = max_rows - len(handles)
        #     handles = handles + [Line2D([0], [0], color='none')] * pad
        #     labels  = labels  + [''] * pad
        #     raw     = raw     + [None] * pad
        #     flat_handles.extend(handles)
        #     flat_labels.extend(labels)
        #     flat_raw.extend(raw)

        # combined_legend = fig.legend(
        #     handles=flat_handles,
        #     labels=flat_labels,
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, -0.00),   # 2x2 is taller, push legend further down
        #     ncol=len(all_columns),
        #     fontsize=10,
        #     edgecolor='#cccccc',
        #     frameon=False,
        #     handlelength=2.0,
        #     handletextpad=0.5,
        #     columnspacing=1.5,
        # )
        fm_in_handles,  fm_in_labels,  fm_in_raw  = group_columns[0]
        fm_out_handles, fm_out_labels, fm_out_raw = group_columns[1]

        merged_fm_column = (
            fm_in_handles + fm_out_handles,
            fm_in_labels  + fm_out_labels,
            fm_in_raw     + fm_out_raw,
        )

        # rebuild group_columns with the two FM groups merged into one column
        group_columns = [merged_fm_column] + group_columns[2:]   # drop old [0] and [1], keep [2:] (PR, BS) unchanged
        head_column = (
            [Line2D([0], [0], color='none'), proxy_mlp, proxy_lin],
            ['Head type', 'MLP head', 'Linear head'],
            ['Head type', None, None],
        )

        all_columns = [head_column] + group_columns   # now 4 columns total instead of 5: head, FM(combined), PR, BS
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
            bbox_to_anchor=(0.5, -0.01),
            ncol=len(all_columns),   # now 4, not 5
            fontsize=10,
            edgecolor='#cccccc',
            frameon=False,
            handlelength=2.0,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        # for text, raw_label in zip(combined_legend.get_texts(), flat_raw):
        #     if raw_label is not None:
        #         text.set_fontweight('bold')
        for text, raw_label in zip(combined_legend.get_texts(), flat_raw):
            if raw_label is not None:
                text.set_fontweight('bold')

        # panel labels A/B/C/D — ax.flat on a 2D array goes row-major: A,B top row; C,D bottom row
        for a, label in zip(ax.flat, 'ABCD'):
            a.text(text_x, text_y, f'{label}', 
                   transform=a.transAxes,
                   fontsize=10, fontweight='bold',
                   ha='left', va='top', clip_on=False)

        fig.subplots_adjust(bottom=0.28, hspace=0.35, wspace=0.05)

        plt.show()
        if save_path:
            fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')

        return fig, ax

# for col, a in enumerate([ax[m, 0], ax[m, 1]]):
#     if col == 1:
#         a.tick_params(labelleft=False)   # keep tick marks, hide numbers
#     else:
#         metric_label = metrics.split('/')[1].upper() if '/' in metrics else metrics.upper()
#         metric_label = metric_label.lower()
#         a.set_ylabel(f'{metric_label_dict[metric_label]}', fontsize=label_fontsize, x=0.045)

#     if metrics == 'test/ibs':
#         if col == 1:
#             a.set_yticks([yticks0[0], yticks0[-1]])   # only first and last tick
#         else:
#             a.set_yticks(yticks0)
#         a.set_ylim(yticks0[0], 0.1)
#     else:
#         if col == 1:
#             a.set_yticks([yticks1[0], yticks1[-1]])
#         else:
#             a.set_yticks(yticks1)
#         a.set_ylim(yticks1[0], 1.0)
#     ...