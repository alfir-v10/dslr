import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def pair_plot(path):
    df = pd.read_csv(path, index_col='Index')
    sns.pairplot(data=df, hue='Hogwarts House')
    plt.tight_layout()
    plt.savefig('pair_plot.png')
    plt.show()


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    pair_plot(pars.parse_args().dataset)
