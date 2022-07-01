import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logreg_train import LogisticRegression


def predict(path_dataset, path_weights):
    df = pd.read_csv(path_dataset, index_col='Index')
    columns = df.columns.tolist()
    df = df.loc[:, columns[0:1] + columns[5:]]
    x_data = df.iloc[:, 1:].values
    x_data = df[['Astronomy', 'Herbology',
                 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
                 'Charms', 'Flying']]  # 'Care of Magical Creatures', 'Arithmancy'
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    model = LogisticRegression().load_weights(path_weights)
    y_pred = model.predict(x_data)
    y_ret = []
    labels = {0: 'Ravenclaw', 1: 'Slytherin', 2: 'Gryffindor', 3: 'Hufflepuff'}
    for y in y_pred:
        y_ret.append(labels[y])
    df = pd.DataFrame(data=zip(range(len(y_ret)), y_ret), columns=['Index', 'Hogwarts House'])
    df.to_csv('houses.csv', index=False)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    pars.add_argument('weights', help='path to weights', type=str)
    predict(pars.parse_args().dataset, pars.parse_args().weights)
