import matplotlib.pyplot as plt

ggplot_styles = {
    'axes.edgecolor': 'grey',
    'axes.facecolor': '#303030',
    'axes.grid': True,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'grey',
    'grid.linewidth': '0.3',
    'xtick.color': 'grey',
    'ytick.color': 'grey',
    'savefig.facecolor': '#303030',
    'figure.facecolor': '#303030',
    'axes.titlecolor': 'white',
}

plt.rcParams.update(ggplot_styles)
