from numpy import *
from matplotlib.pyplot import *
from matplotlib.collections import LineCollection

def text_plotter(x_data, y_data, texts, text_positions, axis,txt_width,txt_height):
    '''plot non-overlapping texts'''
    x1,x2=axis.get_xlim()
    for x,y,text,t in zip(x_data, y_data, texts,text_positions):
        axis.text(x, t+0.25*txt_height, text,rotation=0, color='blue',horizontalalignment='center',verticalalignment='bottom')
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=(x2-x1)*0.02, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)

def text_plot(x_data,y_data,texts,ax,txt_width=0.06,txt_height=0.04):
    '''
    plot non-overlapping texts

    x_data/y_data:
        a set of datas.
    ax:
        axis
    '''
    x1,x2=ax.get_xlim()
    y1,y2=ax.get_ylim()
    #set the bbox for the text. Increase txt_width for wider text.
    txt_height = txt_height*(y2-y1)
    txt_width = txt_width*(x2-x1)
    #Get the corrected text positions, then write the text.
    text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
    text_plotter(x_data,y_data, texts, text_positions, ax, txt_width, txt_height)

def plot_spectrum(el,x=arange(2),offset=[0.,0.],ax=None,lw=3,**kwargs):
    '''
    Plot spectrum.

    el:
        the data.
    x:
        the lower and upper limit of x.
    offset:
        the displace of data.
    ax:
        the ax.
    '''
    N=len(el)
    if ax==None:
        ax=gca()
    #x=repeat([array(x)+offset[0]],N,axis=0).T
    x=array(x)+offset[0]
    el=el+offset[1]
    lc=LineCollection([[(x[0],el[i]),(x[1],el[i])] for i in xrange(N)],**kwargs)
    lc.set_linewidth(lw)
    #pl=ax.plot(x,concatenate([el[newaxis,...],el[newaxis,...]],axis=0))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    #for i in xrange(N):
        #axhline(y=el[i],xmin=x[0],xmax=x[1])
    return ax

