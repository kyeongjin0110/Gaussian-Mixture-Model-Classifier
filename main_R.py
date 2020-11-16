#####################################
##### visualization process gmm #####
#####################################

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

problem = 0
# problem = 1

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

n_classes = len(np.unique(y_train))

print("n_classes")
print(n_classes)
# n_classes = 3

plt_list = []

acc_0 = []
acc_1 = []
acc_2 = []
acc_3 = []

iteration = 20

for n_iter in range (1, iteration+1):
    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=n_classes,
                        covariance_type=covar_type, init_params='kmeans', n_iter=n_iter)) # wc, kmeans, random
                    for covar_type in ['spherical', 'diag', 'tied', 'full']) # full is default

    n_classifiers = len(classifiers)

    # print("n_classifiers")
    # print(n_classifiers)

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
        # print(f"Training time: {stop - start}s")

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

        # if index == 0:
        #     acc_0.append(train_accuracy)
        # elif index == 1:
        #     acc_1.append(train_accuracy)
        # elif index == 2:
        #     acc_2.append(train_accuracy)
        # else:
        #     acc_3.append(train_accuracy)

        y_test_pred = classifier.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                transform=h.transAxes)

        # if index == 0:
        #     acc_0.append(test_accuracy)
        # elif index == 1:
        #     acc_1.append(test_accuracy)
        # elif index == 2:
        #     acc_2.append(test_accuracy)
        # else:
        #     acc_3.append(test_accuracy)

        plt.text(0.05, 0.7, 'Run time: %.6f' % (stop - start),
            transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

        ## visualization decision boundary ##
        hh = .02  # step size in the mesh
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hh),
                            np.arange(y_min, y_max, hh))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max]
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
        plt.axis('off')
        ########################################

        plt_list.append(plt)

    plt.legend(loc='lower right', prop=dict(size=12))

total_acc = [acc_0, acc_1, acc_2, acc_3]
for i in range (0, 4):
    print("acc_{}".format(i))
    for acc in total_acc[i]:
        print(acc)

for _plt in plt_list:
    _plt.show()