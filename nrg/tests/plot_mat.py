#-*-coding:utf-8-*-
from numpy import *
from matplotlib.pyplot import *
from matplotlib import cm
import pdb

myfont = matplotlib.font_manager.FontProperties(
    family='wqy', fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
matplotlib.rcParams["pdf.fonttype"] = 42


def show_arrow():
    N = 5
    arrow = zeros([N, N])
    vals = 0.5**arange(N)
    arrow[0] = arrow[:, 0] = vals
    fill_diagonal(arrow, 0.5 * vals)
    arrow[0, 0] = 0
    ion()
    pcolor(arrow[::-1], cmap=cm.gray_r)
    sq = Rectangle((0, N - 1), 1, 1, facecolor='c',
                   hatch='x', edgecolor='None', alpha=0.5)
    gca().add_patch(sq)
    text(0.5, N - 0.5, '杂质', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0), fontproperties=myfont)
    for i in range(1, N):
        text(i + 0.5, N - 0.5, r'$V_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
        text(0.5, N - 0.5 - i, r'$V_%s^*$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(1, N):
        text(i + 0.5, N - 0.5 - i, r'$\epsilon_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    axis('equal')
    axis('off')
    pdb.set_trace()
    savefig('imp-arrow.pdf')


def show_trid():
    N = 5
    arrow = zeros([N, N])
    vals = sqrt(0.5)**arange(N)
    fill_diagonal(arrow, 0.5 * vals)
    fill_diagonal(arrow[1:, :-1], vals)
    fill_diagonal(arrow[:-1, 1:], vals)
    arrow[0, 0] = 0
    ion()
    pcolor(arrow[::-1], cmap=cm.gray_r)
    sq = Rectangle((0, N - 1), 1, 1, facecolor='c',
                   hatch='x', edgecolor='None', alpha=0.5)
    gca().add_patch(sq)
    text(0.5, N - 0.5, '杂质', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0), fontproperties=myfont)
    text(1.5, N - 0.5, r'$V$', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    text(0.5, N - 1.5, r'$V^*$', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(2, N):
        text(i + 0.5, N - i + 0.5, r'$t_%s$' % (i - 2), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
        text(i - 0.5, N - 0.5 - i, r'$t_%s^*$' % (i - 2), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(1, N):
        text(i + 0.5, N - 0.5 - i, r'$\varepsilon_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    axis('equal')
    axis('off')
    pdb.set_trace()
    savefig('imp-trid.pdf')


def show_arrow_block():
    N = 5
    arrow = zeros([N * 2, N * 2])
    vals = 0.5**arange(N)
    for i in range(2):
        arrow[i, :] = arrow[:, i] = repeat(vals, 2)
    fill_diagonal(arrow, 0.5 * repeat(vals, 2))
    arrow[:2, :2] = 0
    ion()
    pcolor(arrow[::-1], cmap=cm.gray_r, lw=1, edgecolor='#DDDDDD')
    sq = Rectangle((0, N * 2 - 2), 2, 2, facecolor='c',
                   hatch='x', edgecolor='None', alpha=0.5)
    gca().add_patch(sq)
    text(0.5 * 2, (N - 0.5) * 2, '杂质', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0), fontproperties=myfont)
    for i in range(1, N):
        text((i + 0.5) * 2, (N - 0.5) * 2, r'$V_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
        text(0.5 * 2, (N - 0.5 - i) * 2, r'$V_%s^*$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(1, N):
        text((i + 0.5) * 2, (N - 0.5 - i) * 2, r'$\epsilon_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(N - 1):
        xi = yi = i * 2 + 2
        plot([0, 2 * N], [yi, yi], color='#AAAAAA', lw=2)
        plot([xi, xi], [0, 2 * N], color='#AAAAAA', lw=2)
    axis('equal')
    axis('off')
    pdb.set_trace()
    savefig('imp-arrow-block.pdf')


def show_trid_block():
    N = 5
    arrow = zeros([N * 2, N * 2])
    vals = sqrt(0.5)**arange(N)
    for i in range(N):
        i2 = i * 2
        arrow[i2:i2 + 2, i2:i2 + 2] = 0.5 * vals[i]
        if i != N - 1:
            arrow[i2 + 2:i2 + 4, i2:i2 + 2] = vals[i]
            arrow[i2:i2 + 2, i2 + 2:i2 + 4] = vals[i]
            arrow[i2 + 3, i2] = 0
            arrow[i2, i2 + 3] = 0
    arrow[:2, :2] = 0
    ion()
    pcolor(arrow[::-1], cmap=cm.gray_r, lw=1, edgecolor='#DDDDDD')
    sq = Rectangle((0, N * 2 - 2), 2, 2, facecolor='c',
                   hatch='x', edgecolor='None', alpha=0.5)
    gca().add_patch(sq)
    text(1, (N - 0.5) * 2, '杂质', fontsize=16, va='center', ha='center',
         bbox=dict(facecolor='w', edgecolor='none', pad=5.0), fontproperties=myfont)
    for i in range(1, N):
        text((i + 0.5) * 2, (N - i + 0.5) * 2, r'$T_%s$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
        text((i - 0.5) * 2, (N - 0.5 - i) * 2, r'$T_%s^*$' % (i - 1), fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(1, N):
        text((i + 0.5) * 2, (N - 0.5 - i) * 2, r'$E_%s$' % i, fontsize=16, va='center', ha='center',
             bbox=dict(facecolor='w', edgecolor='none', pad=5.0))
    for i in range(N - 1):
        xi = yi = i * 2 + 2
        plot([0, 2 * N], [yi, yi], color='#AAAAAA', lw=2)
        plot([xi, xi], [0, 2 * N], color='#AAAAAA', lw=2)
    axis('equal')
    axis('off')
    pdb.set_trace()
    savefig('imp-trid-block.pdf')


# show_arrow_block()
show_trid_block()
# show_arrow()
# show_trid()
