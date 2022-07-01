import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import random
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self, alpha=0.01, n_iter=10000):
        self.cost = []
        self.w = []
        self.alpha = alpha
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def cost_func(self, y_pred, y, m):
        return (1 / m) * (-(np.sum(y.T.dot(np.log(y_pred)) + (1 - y).T.dot(np.log(1 - y_pred)))))

    def gradient_descent(self, x, y_pred, w, y, m):
        return w - (self.alpha / m) * np.dot(x.T, (y_pred - y))

    def fit(self, x, y, grad_desc_mode='SGD', batch_size=16, shuffle=True, info=True):
        x = np.insert(x, 0, 1, axis=1)
        m = len(y)

        for i in np.unique(y):
            y_labels = np.where(y == i, 1, 0)
            w = np.zeros(x.shape[1])
            cost = []
            for num_iter in range(self.n_iter):

                if shuffle:
                    lst = list(zip(x, y_labels))
                    np.random.shuffle(lst)
                    x, y_labels = zip(*lst)
                    x = np.array(x)
                    y_labels = np.array(y_labels)

                y_pred = self.sigmoid(x.dot(w))
                if grad_desc_mode == 'BGD':
                    '''Batch Gradient Descent'''
                    w = self.gradient_descent(x=x, y_pred=y_pred, w=w, y=y_labels, m=m)
                    # cost.append(self.cost_func(y_pred=y_pred, y=y_labels, m=m))

                if grad_desc_mode == 'MBSGD':
                    ''' Mini-batch Stochastic Gradient Descent '''
                    assert 1 <= batch_size < x.shape[0], 'Batch Size must be greater than 1 or equal 1'
                    n_batches = x.shape[0] // batch_size
                    x_batches = np.array_split(x, n_batches)
                    y_labels_batches = np.array_split(y_labels, n_batches)
                    for n, (x_batch, y_batch) in enumerate(zip(x_batches, y_labels_batches)):
                        if info:
                            print(f'{i} vs All # n_iter: {num_iter} # n_batch: {n}/{len(x_batches)}')
                        y_pred_batch = self.sigmoid(x_batch.dot(w))
                        w = self.gradient_descent(x=x_batch, y_pred=y_pred_batch, w=w, y=y_batch, m=batch_size)
                    cost.append(self.cost_func(y_pred=y_pred, y=y_labels, m=m))

                # TODO: early stopping
                
                # if self.cost_func(y_pred=y_pred, y=y_labels, m=m)
                # if grad_desc_mode == 'SGD':
                #     '''Stochastic Gradient Descent'''
                #     for n in range(x.shape[0]):
                #         if info:
                #             print(f'{i} vs All # n_iter: {num_iter} # n_sample: {n}/{x.shape[0]}')
                #         w = self.gradient_descent(x[n, :], y_pred[n], w, y_labels[n])
                #     cost.append(self.cost_func(y_pred=y_pred, y=y_labels, m=m))

            # self.cost.append((cost, i))
            self.w.append((w, i))
        return self

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        y_pred = [max((self.sigmoid(i.dot(w)), c) for w, c in self.w)[1] for i in x]
        return y_pred

    def accuracy_score(self, x, y):
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


def standart_scaler(x):
    return (x - x.mean()) / x.std()


def train(path):
    df = pd.read_csv(path, index_col='Index')
    columns = df.columns.tolist()
    df = df.loc[:, columns[0:1] + columns[5:]]
    classes = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

    df['Hogwarts House'].replace(classes, inplace=True)
    df.dropna(inplace=True)
    y_data = df.iloc[:, 0].values
    x_data = df[['Astronomy', 'Herbology',
                 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
                 'Charms', 'Flying']]  # 'Care of Magical Creatures', 'Arithmancy'
    x_data = x_data.values
    x_data = standart_scaler(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.3,
                                                        random_state=21)

    model = LogisticRegression(alpha=0.1, n_iter=7000).fit(x_train, y_train, grad_desc_mode='BGD',
                                                            batch_size=1,
                                                            shuffle=False,
                                                            info=False)
    model.save_weights('weights.csv')
    score = model.accuracy_score(x_test, y_test)
    print('Score model ', score)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='path to dataset', type=str)
    train(pars.parse_args().dataset)
    # train('datasets/dataset_train.csv')
