import seaborn as sns
import matplotlib as plt
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from shutil import copy2

sns.set_context('paper');
curdir = os.path.abspath(os.path.dirname(__file__))
plt.style.use(['seaborn-ticks', os.path.join(curdir, 'thesisplot.style')])

def maincolor():
    return '#ed6751'

def label_subplots(*axes, labels='ABCDEFGHIJKLMNOPQRSTUVWXYZ', tmpl='{}'):
    for i, ax in enumerate(axes):
        ax.set_title(tmpl.format(labels[i]))
        ax.title.set_position([0,1.05])
        ax.title.set_ha('left')

from matplotlib.ticker import FuncFormatter
thousands_ticks = FuncFormatter(lambda x, pos: '{0:g}k'.format(x/1000))


def cm2inch(*args):
    return list(map(lambda x: x/2.54, args))


def restyle_title(*axes, 
        labels='abcdefghijklmnopqrstuvwxyz'.upper(), 
        label_tmpl = '{label}.',
        title_tmpl='{title}',
        pad=5, x=0, y=1.07,
        label=True):
    
    if label:
        title_tmpl = ' '*pad + title_tmpl
    
    titles = []
    for i, ax in enumerate(axes):
        
        # Label
        if label:
            lbl = ax.text(x, y,
                label_tmpl.format(label=labels[i]), 
                transform=ax.transAxes, 
                verticalalignment='center',
                weight='bold',
                fontsize=ax.title.get_fontsize())
        
        # Title
        txt = ax.text(x, y,
                title_tmpl.format(title=ax.get_title()), 
                transform=ax.transAxes, 
                verticalalignment='center',
                fontsize=ax.title.get_fontsize())
        
        if label:
            titles.append((lbl,txt))
        else:
            titles.append(txt)
        
        # Hide title
        ax.set_title('')

    return titles

def CustomCmap(from_rgb, to_rgb, via_rgb=None):
    """
    A custom color map, linearly from one color two another 
    (possibly via a third)
    """

    # from color r,g,b
    if type(from_rgb) == str:
        r1,g1,b1 = get_color(from_rgb, 'rgb')
    else:
        r1,g1,b1 = from_rgb

    # to color r,g,b
    if type(to_rgb) == str:
        r2,g2,b2 = get_color(to_rgb, 'rgb')
    else:   
        r2,g2,b2 = to_rgb
    
    if via_rgb:
        if type(via_rgb) == str:
            rv,gv,bv = get_color(via_rgb, 'rgb')
        else:
            rv,gv,bv = via_rgb
        cdict = {'red': ((0, r1, r1),
                         (.5, rv,rv),
                         (1, r2, r2)),
               'green': ((0, g1, g1),
                         (.5, gv,gv),
                         (1, g2, g2)),
               'blue': ((0, b1, b1),
                        (.5, bv,bv),
                        (1, b2, b2))}
    else:
        cdict = {'red': ((0, r1, r1),
                       (1, r2, r2)),
               'green': ((0, g1, g1),
                        (1, g2, g2)),
               'blue': ((0, b1, b1),
                       (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def get_all_colors() :
    return {
        'orange': {
            'rgb': [237/255, 103/255, 81/255],
            'hex': '#ed6751'
        },
        'blue': {
            'rgb': [117/255, 195/255,209/255],
            'hex': '#78C3D0'
        },
        'yellow': {
            'rgb': [255/255,241/255,150/255],
            'hex': '#E5D472'
        },
        'white': {
            'rgb': [1,1,1],
            'hex': '#ffffff'
        }
    }

def get_color(name, type='rgb'):
    return get_all_colors()[name][type]

def copyfig(src, target_dir='/Users/Bas/thesis/figures'):
    dst = os.path.join(os.path.realpath(target_dir), src)
    copy2(src, dst)