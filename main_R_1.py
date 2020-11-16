##########################################
##### for check number of components #####
##########################################

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import time


def make_ellipses(gmm, ax):
    for n, color in enumerate('rg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

# problem = 0
problem = 1

if problem == 0:
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p1_train_input.txt') as f:   
        lines = f.readlines()
        X_train = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p1_train_target.txt') as f:
        lines = f.readlines()
        y_train = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p1_test_input.txt') as f:
        lines = f.readlines()
        X_test = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p1_test_target.txt') as f:
        lines = f.readlines()
        y_test = np.genfromtxt(lines)
else:
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p2_train_input.txt') as f:    
        lines = f.readlines()
        X_train = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p2_train_target.txt') as f:
        lines = f.readlines()
        y_train = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p2_test_input.txt') as f:
        lines = f.readlines()
        X_test = np.genfromtxt(lines)
    with open('/home/icirc/Desktop/KJ/Class_Project/Machine_Learning_3/data/p2_test_target.txt') as f:
        lines = f.readlines()
        y_test = np.genfromtxt(lines)

X_total = np.concatenate((X_train, X_test), axis=0)
y_total = np.concatenate((y_train, y_test), axis=0)

target_names = ['0', '1']

## check number of components
n_comps = np.arange(1, 10)
clfs = [GMM(n, n_iter = 1000).fit(X_train) for n in n_comps]
bics = [clf.bic(X_train) for clf in clfs]
aics = [clf.aic(X_train) for clf in clfs]

plt.plot(n_comps, bics, label = 'BIC')
plt.plot(n_comps, aics, label = 'AIC')
plt.xlabel('n_components')
plt.legend()
plt.show()
################################

n_classes = len(np.unique(y_train))

print("n_classes")
print(n_classes)
# n_classes = 3

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20)) # wc, kmeans, random
                for covar_type in ['spherical', 'diag', 'tied', 'full']) # full is default

n_classifiers = len(classifiers)

print("n_classifiers")
print(n_classifiers)

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                for i in xrange(n_classes)])

    start = time.time()
    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rg'):
        data = X_total[y_total == n]
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rg'):
        data = X_test[y_test == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    plt.text(0.05, 0.7, 'Run time: %.6f' % (stop - start),
            transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(loc='lower right', prop=dict(size=12))
plt.show()

