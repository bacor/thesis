import seaborn as sns
import matplotlib as plt
import os

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
