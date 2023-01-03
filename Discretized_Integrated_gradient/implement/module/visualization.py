import matplotlib as mpl
from matplotlib.colors import Normalize, rgb2hex
from IPython.display import HTML
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np

def hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    
    cmap_bound = torch.max(torch.abs(attrs))
    norm = Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: rgb2hex(cmap(norm(x))), attrs))
    return colors

def print_html_language(input,att,index2word):
    colors = colorize(att.to('cpu').detach())
    words=[index2word[i] for i in input.to('cpu').numpy()]
    HTML("".join(list(map(hlstr, words, colors))))

def print_top_5_words(input,att,index2word):
    words=[index2word[i] for i in input.to('cpu').numpy()]
    print(f"\nTop 5 Important words: "
        f"{[words[i] for i in torch.argsort(att.to('cpu').detach(),descending=True)[:5]]}\n")

def get_color(attr):
    if attr > 0:
        g = int(128*attr) + 127
        b = 128 - int(64*attr)
        r = 128 - int(64*attr)
    else:
        g = 128 + int(64*attr)
        b = 128 + int(64*attr)
        r = int(-128*attr) + 127
    return r,g,b

def print_att_sentence(input,id2word,color):
    sentence=''
    for i,c in zip(input.numpy(),color):
        if i!=0:
            word=id2word[i]
            sentence=sentence+"\033[38;2;{};{};{}m {} \033[0m".format(c[0],c[1],c[2],word)

    print(sentence)