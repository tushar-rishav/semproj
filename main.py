# -*- coding: utf-8 -*-
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from itertools import cycle
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



class Data(object):
    """
        Class for pre-processing data.
    """
    def __init__(self, filename="data.csv", cold=0):
        self.label = ["Rel Compactness", "Surface Area", "Wall Area", "Roof Area",
                      "Overall Ht.", "Orientation", "Glazing Area",
                      "Glazing Area Distribution", "Heating Load", "Cooling Load"]
        self.data = pd.read_csv(filename)
        self.cold = cold
        self.X, self.y = self.data.ix[:,0:8].as_matrix(), self.data.ix[:,8:10].as_matrix()
        scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.4,
                                                                                random_state=0)
        scaler.fit(self.X_train)
        scaler.fit(self.X)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X = scaler.transform(self.X)
    
    def get_sp_rank(self):
        """
            Return Spearman rank-order correlation coefficient and p-value
            to test for non-correlation.
        """
        
        return [(spearmanr(self.data.ix[:, i:i+1], self.data.ix[:, 8:9]),
                 spearmanr(self.data.ix[:, i:i+1], self.data.ix[:, 9:10]))
                 for i in range(len(self.label)-2)]



def cross_validator(func):
    def inner_wrap(self):
        est, est_cv = func()
        X = self.X
        y = self.y
        alphas = np.logspace(-4, -0.5, 30)
        scores = []
        scores_std = []
        n_folds = 5

        for alpha in alphas:
            est.alpha = alpha
            this_scores = cross_val_score(est, X, y, cv=n_folds, n_jobs=1)
            scores.append(np.mean(this_scores))
            scores_std.append(np.std(this_scores))

        scores, scores_std = np.array(scores), np.array(scores_std)
        est_cv.alphas = alphas
        est_cv.random_state = 0
        k_fold = KFold(n_folds)
        scores = []
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            est_cv.fit(X[train, 0:8], y[train, self.cold: 1+self.cold])
            scores.append(est_cv.score(X[test, 0:8],
                                    y[test, self.cold: 1+self.cold]))

        return np.mean(scores)*100, est_cv
    
    return inner_wrap


class Ridge_M(Data, object):
    """
        Ridge regression model.
    """
    def __init__(self):
        Data.__init__(self)
        self.acc, self.est_cv = Ridge_M.fit(self)
    
    @staticmethod
    @cross_validator
    def fit():
        est = linear_model.Ridge()
        est_cv = linear_model.RidgeCV()
        return est, est_cv

    @property
    def accuracy(self):
        return self.acc
    
    def get_prediction(self):
        return self.est_cv.predict(self.X)


class ANN(Data, object):
    def __init__(self):
        Data.__init__(self)
        self.reg = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=(5, 2), random_state=1)
        self.reg.fit(self.X_train, self.y_train[:,self.cold:self.cold+1].ravel())
    
    @property
    def weights(self):
        return self.reg.coefs_

    @property
    def accuracy(self):
        return self.reg.score(self.X_test, self.y_test[:,self.cold:self.cold+1].ravel())*100
    
    def get_prediction(self):
        return self.reg.predict(self.X_test)


# PLOT GRAPHS

def plot_model_accuracy(models):
    X = range(len(models))
    Y = map(lambda m: m[1].accuracy, models)
    cycol = cycle('bgrcmk').next

    for i in range(len(Y)):
        plt.bar(i, Y[i], 0.5, align='center', color=cycol())

    for x, y in zip(X,Y):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    
    labels = [m[0] for m in models]
    plt.xticks(X, labels)
    plt.xlabel('Regression models')
    plt.title('Accuracy (%)')
    plt.ylim(80,100)
    plt.show()


def plot_feature_correlation(data):
    """
        Scatter plot of X vs X and Y vs X.

        X1	Relative Compactness 
        X2	Surface Area 
        X3	Wall Area 
        X4	Roof Area 
        X5	Overall Height 
        X6	Orientation 
        X7	Glazing Area 
        X8	Glazing Area Distribution
        y1	Heating Load 
        y2	Cooling Load
    """
    
    cycol = cycle('bgrcmk').next
    nf = len(data.label) - 2
    
    plt.figure(1)   # X vs X
    for i in range(nf):
        for j in range(nf):
            plt.subplot(nf, nf, i*nf+j+1)
            plt.axis('off')
            plt.scatter(data.data.ix[:, j:j+1].as_matrix(),
                        data.data.ix[:, i:i+1].as_matrix(),
                        color = cycol())

    plt.figure(2)   # Y vs X
    for i in range(nf):
        clr = cycol()
        plt.subplot(2, 8, i+1)
        plt.axis('off')
        plt.scatter(data.data.ix[:, i:i+1].as_matrix(),
                    data.data.ix[:, nf:nf+1].as_matrix(),
                    color = clr)
        
        plt.subplot(2, 8, i+9)
        plt.axis('off')
        plt.scatter(data.data.ix[:, i:i+1].as_matrix(),
                    data.data.ix[:, nf+1:nf+2].as_matrix(),
                    color = clr)
    
    plt.show()


def plot_pd(data):
    """
        Plot probablility densities.
    """
    cycol = cycle('bgrcmk').next
    nf = len(data.label)

    for i in range(nf):
        plt.figure(i+1)
        pdf, bins, patches = plt.hist(data.data.ix[:, i:i+1].as_matrix(),
                                      20, normed=1, facecolor=cycol(), alpha=0.75)
        print np.sum(pdf * np.diff(bins))   # sums to 1
        
        plt.xlabel(data.label[i])
        plt.ylabel('Probability')
        plt.grid(True)

        plt.show()


def plot_spearman(sp_result, data):
    print [x[0].correlation for x in sp_result]
    plt.bar(range(len(sp_result)), [x[0].correlation for x in sp_result], tick_label=data.label[:8], align="center")
    plt.title("Spearman correlation coefficient")
    plt.show()

def get_cov_matrix(data):
    """
        Compute the covariance matrix for the dataset
    """
    nf = len(data.label)-2
    cov_matrix = []
    for i in range(nf):
        cov = []
        for j in range(nf):
            cov.append(spearmanr(data.data.ix[:, i:i+1].as_matrix(),
                        data.data.ix[:, j:j+1].as_matrix()).correlation)
        cov_matrix.append(cov)
    
    return cov_matrix

def plot_cov_matrix(cov_matrix):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Color Map: Covariance matrix')
    plt.imshow(cov_matrix)
    ax.set_aspect('equal')
    labels = ["X"+str(i) for i in range(9)]
    labels.extend(["y1", "y2"])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    cax = fig.add_axes([0,0.1,0.95,0.80])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

def plot_output_target_graph(y_output_hl, y_output_cl, y_target):
    plt.figure(1)
    X = range(len(y_target[:, 0:1]))
    plt.plot(X, y_target[:, 0:1], 'b+', label='Target HL')
    plt.plot(X, y_output_hl, 'ro', label='Output HL')
    plt.title("Target HL vs Output HL")
    plt.xlabel("Sample number")
    plt.ylabel("Responses")
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(X, y_target[:, 1:2], 'b+', label='Target CL')
    plt.plot(X, y_output_cl, 'ro', label='Output CL')
    plt.title("Target CL vs Output CL")
    plt.xlabel("Sample number")
    plt.ylabel("Responses")
    plt.legend(loc='best')
    plt.show()


def main():
    data = Data(cold=0)
    ridge = Ridge_M()
    ann = ANN()
    models = [("Ridge", ridge), ("ANN", ann)]
    
    for model_name, model_obj in models: print model_name, model_obj.accuracy
    sp_result = data.get_sp_rank()
    cov_matrix = get_cov_matrix(data)
    for el in cov_matrix:
        print el
    plot_cov_matrix(cov_matrix)
    plot_feature_correlation(data)
    plot_model_accuracy(models)
    plot_pd(data)
    plot_spearman(sp_result, data)
    
    ann_y_output_hl, ridge_y_output_hl = ann.get_prediction(), ridge.get_prediction()
    data.cold = 1
    ann = ANN()
    ridge = Ridge_M()
    _ = ridge.accuracy
    ann_y_output_cl, ridge_y_output_cl = ann.get_prediction(), ridge.get_prediction()
    plot_output_target_graph(ann_y_output_hl, ann_y_output_cl, data.y_test)
    plot_output_target_graph(ridge_y_output_hl, ridge_y_output_cl, data.y)
    
    

if __name__ == "__main__":
    main()
