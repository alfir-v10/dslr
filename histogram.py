import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def histogram(path):
    df = pd.read_csv(path, index_col='Index')
    skill_columns = df.select_dtypes(include=['float64']).columns

    fig, ax = plt.subplots(7, 2, figsize=(15, 25))
    fig.suptitle('Hogwarts Courses Distributions between all houses',
                 horizontalalignment='center', verticalalignment='bottom')
    fig.delaxes(ax[6][1])
    j = 0
    i = 0
    for skill in skill_columns:
        if i % 2 == 0 and i != 0:
            j += 1
            i = 0
        sns.histplot(ax=ax[j, i], data=df, x=skill, hue='Hogwarts House', kde=True)
        i += 1
    plt.tight_layout()
    plt.savefig('histplot.png')
    plt.show()


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    histogram(pars.parse_args().dataset)
