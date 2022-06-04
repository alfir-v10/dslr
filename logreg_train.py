import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:

    def __init__(self, alpha=0.01, n_iter=100):
        self.cost = []
        self.w = []
        self.alpha = alpha
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_func(self, y_pred, w, y, m):
        return (1 / m) * (np.sum(-y.T.dot(np.log(y_pred)) - (1 - y).T.dot(np.log(1 - y_pred))))

    def gradient_descent(self, x, y_pred, w, y, m):
        return w - (self.alpha / m) * np.dot(x.T, (y_pred - y))

    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y):
            y_labels = np.where(y == i, 1, 0)
            w = np.zeros(x.shape[1])
            cost = []
            for _ in range(self.n_iter):
                y_pred = self.sigmoid(x.dot(w))
                w = self.gradient_descent(x=x, y_pred=y_pred, w=w, y=y_labels, m=m)
                cost.append(self.cost_func(y_pred=y_pred, w=w, y=y_labels, m=m))
            self.cost.append((cost, i))
            self.w.append((w, i))
        return self

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        y_pred = [max((self.sigmoid(i.dot(w)), c) for w, c in self.w)[1] for i in x]
        return y_pred

    def score(self, x, y):
        return 1 / len(y) * sum(self.predict(x) == y)

    def save_weights(self, path):
        w = np.array([list(k[0]) + [k[1]] for k in self.w])
        df = pd.DataFrame(data=w, columns=list(range(w.shape[1])))
        df.to_csv(path)

    def load_weights(self, path):
        w = pd.read_csv(path)
        for i in range(len(w)):
            arr = w.iloc[i, 1:-1].to_numpy(), w.iloc[i, -1]
            self.w.append(arr)
        return self


def train(path):
    df = pd.read_csv(path, index_col='Index')
    columns = df.columns.tolist()
    df = df.loc[:, columns[0:1] + columns[5:]]
    classes = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

    df['Hogwarts House'].replace(classes, inplace=True)
    df.dropna(inplace=True)
    y_data = df.iloc[:, 0].values
    x_data = df.iloc[:, 1:].values
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33)
    model = LogisticRegression().fit(x_train, y_train)
    model.save_weights('weights.csv')
    score = model.score(x_test, y_test)
    print('Score model ', score)
    print(x_data.shape)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    train(pars.parse_args().dataset)
