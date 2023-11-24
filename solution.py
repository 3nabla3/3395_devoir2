import numpy as np
import pandas as pd
from tqdm import trange
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
        # onsidérant un minibatch d’exemples,
        # cette fonction devrait calculer le gradient de la fonction de perte
        # par rapport au paramètre w. Les entrées de la fonction sont X
        # (un tableau numpy de dimension (minibatch size, 17)) et y (un
        # tableau numpy de dimension (minibatch size, 3)) et la sortie de-
        # vrait être le gradient calculé, un tableau numpy de dimension
        # (17, 3), soit la même dimension que celle du paramètre w

        # ret = np.zeros_like(self.w)

        # for j in range(self.w.shape[1]):
        #     for k in range(self.w.shape[0]):
        #         a = 4 * (x[:, k] @ y[:, j])
        #         b = 2 * (x @ self.w[:, j]) * x[:, k]
        #         c = b - a

        #         condition = 2 - (x @ self.w[:, j]) * y[:, j] <= 0

        #         ret[k, j] = np.sum(np.where(condition, 0, c))

        # return ret

        ret = np.zeros_like(self.w)

        x_dot_y = x.T @ y  # Precompute x transpose multiplied by y

        for j in range(self.w.shape[1]):
            x_dot_w_j = x @ self.w[:, j]
            b = 2 * x_dot_w_j * x.T

            a = 4 * x_dot_y[:, j]
            c = b - a[:, np.newaxis]

            condition = 2 - x_dot_w_j * y[:, j] <= 0
            ret[:, j] = np.sum(
                np.where(condition[:, np.newaxis], 0, c.T),
                axis=0
            )

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

    def fit(self, x_train, y_train, x_test, y_test):
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

        for iteration in trange(self.niter):
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
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=100, verbose=False)
    train_losses, train_accs, test_losses, test_accs = svm.fit(
        x_train, y_train, x_test, y_test
    )

    # # to infer after training, do the following:
    y_inferred = svm.infer(x_test)

    # to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(
        y_train, 3
    )  # one-versus-all labels
    y_test_ova = svm.make_one_versus_all_labels(
        y_test, 3
    )  # one-versus-all labels

    accuracy = svm.compute_accuracy(y_inferred, y_test_ova)
    print(f'{accuracy = }')

    # svm.w = np.zeros([x_train.shape[1], 3])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
    # print(f'{loss = }')
