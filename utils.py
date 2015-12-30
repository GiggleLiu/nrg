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


