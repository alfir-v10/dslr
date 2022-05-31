import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def scatter_plot(path):
    df = pd.read_csv(path, index_col='Index')
    g = sns.PairGrid(df, hue='Hogwarts House')
    g.map(sns.scatterplot)
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    plt.show()

if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    scatter_plot(pars.parse_args().dataset)
