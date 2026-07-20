

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import ImageOps
import matplotlib as mpl
from utils import linear_acc
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

            f_plot_title = f'{plot_title} acc. = {acc1:.1f}% auc. = {auc1:.1f}'
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
