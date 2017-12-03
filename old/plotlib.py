from numpy import *
from matplotlib.pyplot import *
from matplotlib.collections import LineCollection


def text_plotter(x_data, y_data, texts, text_positions, axis, txt_width, txt_height):
    '''plot non-overlapping texts'''
    x1, x2 = axis.get_xlim()
    for x, y, text, t in zip(x_data, y_data, texts, text_positions):
        axis.text(x, t + 0.25 * txt_height, text, rotation=0, color='blue',
                  horizontalalignment='center', verticalalignment='bottom')
        if y != t:
            axis.arrow(x, t, 0, y - t, color='red', alpha=0.3, width=txt_width * 0.1,
                       head_width=(x2 - x1) * 0.02, head_length=txt_height * 0.5,
                       zorder=0, length_includes_head=True)


def text_plot(x_data, y_data, texts, ax, txt_width=0.06, txt_height=0.04):
    '''
    plot non-overlapping texts

    Parameters:
        :x_data/y_data: 1d array, a set of datas.
        :ax: axis
    '''
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    # set the bbox for the text. Increase txt_width for wider text.
    txt_height = txt_height * (y2 - y1)
    txt_width = txt_width * (x2 - x1)
    # Get the corrected text positions, then write the text.
    text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
    text_plotter(x_data, y_data, texts, text_positions,
                 ax, txt_width, txt_height)


def plot_spectrum(el, x=arange(2), offset=[0., 0.], ax=None, lw=3, **kwargs):
    '''
    Plot spectrum.

    Parameters:
        :el: 1d array, data.
        :x: len-2 tuple, lower and upper limit of x.
        :offset: len-2 tuple, displacement.
        :ax,lw: axis, line width
    '''
    N = len(el)
    if ax == None:
        ax = gca()
    x = array(x) + offset[0]
    el = el + offset[1]
    lc = LineCollection([[(x[0], el[i]), (x[1], el[i])]
                         for i in xrange(N)], **kwargs)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    return ax


def show_scale(scale, hspace=1., offset=(0, 0), **kwargs):
    '''
    show scale.
    '''
    scale_visual = concatenate(
        [-scale._data_neg, scale._data[::-1]]) + offset[0]
    yi = offset[1]
    plot([scale_visual[0], scale_visual[-1]], [yi, yi], 'k', lw=1)
    # scatter(scale_visual[:,i],ones(scale.N*2+2)*yi,s=30,color='k',**kwargs)
    plot(scale_visual, ones(scale.N * 2 + 2) * yi, marker='$\\bf |$', **kwargs)


def show_binner(binner, lw=1, **kwargs):
    '''
    show datas.

    Parameters:
        :binner: <Binner>,
        :lw: number, the line width.
        **kwargs, key word arguments for LineCollection.
    '''
    bins = binner.bins
    ax = gca()
    lc = LineCollection([[(bins[i], 0), (bins[i], binner.weights[i])]
                         for i in xrange(binner.N)], **kwargs)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
