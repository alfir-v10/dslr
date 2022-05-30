import argparse
import pandas as pd
import numpy as np
from datetime import datetime


def get_count(array):
    return array.shape[0]


def get_mean(array):
    return array.sum() / array.shape[0]


# TODO: diffs
def get_std(array):
    return np.sqrt(np.sum(np.power(array - get_mean(array), 2)) / array.shape[0])


# TODO: diffs
def get_quartile(array, perc):
    array = np.sort(array)
    per = (perc / 100) * array.shape[0]
    if per % 1.0:
        return array[int(per)]
    else:
        return array[int(per) - 1]


def get_min(array):
    min_arr = array[0]
    for arr in array:
        if arr < min_arr:
            min_arr = arr
    return min_arr


def get_max(array):
    max_arr = array[0]
    for arr in array:
        if arr > max_arr:
            max_arr = arr
    return max_arr


#hard way without pd.read_csv
def load_data(path):
    with open(path, 'r') as f:
        lines = [line.rstrip().split(',') for line in f.readlines()]
    print(lines[:5])
    columns_name = lines[0]
    lines = [[int(line[0]), str(line[1]), str(line[2]), str(line[3]),
              datetime.strptime(line[4], '%Y-%m-%d'),
              str(line[4])] + [float(k) if k != '' else np.NAN for k in line[6:]] for line in lines[1:]]
    return [columns_name] + lines


def describe(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    print(df.head())
    print(df.describe())
    print(df.columns)
    statistics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    df_numerical = df.iloc[:, 6:]
    statistics_dict = dict().fromkeys(df_numerical.columns, [])

    for col in df_numerical.columns:
        for stat in statistics:
            if stat in ['25%', '50%', '75%']:
                tmp = globals()['get_quartile'](df_numerical[col], int(stat.rstrip('%')))
                statistics_dict[col] = statistics_dict[col] + [tmp]
            else:
                tmp = globals()[f'get_{stat}'](df_numerical[col])
                statistics_dict[col] = statistics_dict[col] + [tmp]
    for k, v in statistics_dict.items():
        print(k, v)
    df_new = pd.DataFrame(data=statistics_dict, index=statistics)
    # df_describe = df.iloc[:, 1:].describe()
    print(df_new)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    describe(pars.parse_args().dataset)