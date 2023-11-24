import numpy as np
import pandas as pd
from tqdm import trange
from tqdm.contrib.concurrent import process_map

import pylab as pl
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        res = np.full((y.shape[0], m), -1)
        res[np.arange(y.shape[0]), y] = 1
        return res

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        l_terms = np.maximum(0, 2 - x @ self.w * y) ** 2
        hinge_loss = np.mean(np.sum(l_terms, axis=1))

        w_norm_sq = np.sum(self.w ** 2, axis=1)
        l2_reg = self.C / 2 * np.sum(w_norm_sq)

        return hinge_loss + l2_reg

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        ret = np.zeros_like(self.w)

        for j in range(self.w.shape[1]):
            dot_prod = np.dot(x, self.w[:, j])
            condition = 2 - dot_prod * y[:, j] <= 0
            hinge_grad = np.where(
                condition[:, np.newaxis], 0, (2 * dot_prod - 4 * y[:, j])[:, np.newaxis] * x)
            total = np.sum(hinge_grad, axis=0)

            l2_grad = self.C * self.w[:, j]
            ret[:, j] = total / x.shape[0] + l2_grad

        return ret

    # Batcher function

    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size

        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        probs = x @ self.w
        chosen = np.argmax(probs, axis=1)
        return self.make_one_versus_all_labels(chosen, self.m)

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        return np.mean(np.all(y_inferred == y, axis=1))

    def fit(self, x_train, y_train, x_test, y_test, pos=0):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, num_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in trange(self.niter, position=pos):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Star_classification/"
    dataset = pd.read_csv(data_path + "star_classification.csv")
    y = dataset['class']
    x = dataset.drop(['class', 'rerun_ID'], axis=1)

    # we replace the dataset class with a number (the class are : 'GALAXY' 'QSO' 'STAR')
    y = y.replace('GALAXY', 0)
    y = y.replace('QSO', 1)
    y = y.replace('STAR', 2)

    # split dataset in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=40)

    # convert sets to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print(f'{x_train.shape = }')
    print(f'{y_train.shape = }')
    print(f'{x_test.shape = }')
    print(f'{y_test.shape = }')

    print("Fitting the model...")

    train_losses_list, train_accs_list, test_losses_list, test_accs_list = [], [], [], []

    def fit(arg):
        pos, c = arg
        svm = SVM(eta=0.0001, C=c, niter=200, batch_size=100, verbose=False)
        return svm.fit(
            x_train, y_train, x_test, y_test, pos=pos
        )

    # calculate the metrics in parallel
    train_losses_list, train_accs_list, test_losses_list, test_accs_list = \
        zip(*process_map(fit, enumerate([1, 5, 10]), max_workers=3))

    print("Plotting...")

    pl.plot(train_losses_list[0], label='C = 1')
    pl.plot(train_losses_list[1], label='C = 5')
    pl.plot(train_losses_list[2], label='C = 10')
    pl.xlabel("Iteration")
    pl.ylabel("Train Loss")
    pl.legend()
    pl.savefig("images/train_loss.png")
    pl.clf()

    pl.plot(train_accs_list[0], label='C = 1')
    pl.plot(train_accs_list[1], label='C = 5')
    pl.plot(train_accs_list[2], label='C = 10')
    pl.xlabel("Iteration")
    pl.ylabel("Train Accuracy")
    pl.legend()
    pl.savefig("images/train_acc.png")
    pl.clf()

    pl.plot(test_losses_list[0], label='C = 1')
    pl.plot(test_losses_list[1], label='C = 5')
    pl.plot(test_losses_list[2], label='C = 10')
    pl.xlabel("Iteration")
    pl.ylabel("Test Loss")
    pl.legend()
    pl.savefig("images/test_loss.png")
    pl.clf()

    pl.plot(test_accs_list[0], label='C = 1')
    pl.plot(test_accs_list[1], label='C = 5')
    pl.plot(test_accs_list[2], label='C = 10')
    pl.xlabel("Iteration")
    pl.ylabel("Test Accuracy")
    pl.legend()
    pl.savefig("images/test_acc.png")
    pl.clf()
