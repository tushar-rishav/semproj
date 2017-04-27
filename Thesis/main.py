# -*- coding: utf-8 -*-
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
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
        self.label = ["Rel Compactness", "Surface Area", "Wall Area", 
                      "Roof Area", "Overall Ht.", "Orientation", 
                      "Glazing Area", "Glazing Area Distribution", 
                      "Heating Load", "Cooling Load"]
        self.data = pd.read_csv(filename)
        self.cold = cold
        X = self.data.ix[:,0:8].as_matrix()
        y = self.data.ix[:,8:10].as_matrix()
        scaler = StandardScaler()
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
                                        X,y, test_size=0.3, random_state=0)
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
    
    def get_sp_rank(self):
        """
            Return Spearman rank-order correlation coefficient and p-value
            to test for non-correlation.
        """
        
        return [(spearmanr(self.data.ix[:, i:i+1], self.data.ix[:, 8:9]),
                 spearmanr(self.data.ix[:, i:i+1], self.data.ix[:, 9:10]))
                 for i in range(len(self.label)-2)]



def cross_validator(func):
    def inner_wrap(self, data):
        est, est_cv = func()
        X = data.ix[:,0:8].as_matrix()
        y = data.ix[:,8:10].as_matrix()
        alphas = np.logspace(-4, -0.5, 30)
        scores = []
        scores_std = []
        n_folds = 5

        for alpha in alphas:
            est.alpha = alpha
            this_scores = cross_val_score(est, X, y,
                            cv=n_folds, n_jobs=1)
            scores.append(np.mean(this_scores))
            scores_std.append(np.std(this_scores))

        scores, scores_std = np.array(scores), np.array(scores_std)

        plt.figure().set_size_inches(8, 6)
        plt.semilogx(alphas, scores)

        std_error = scores_std / np.sqrt(n_folds)

        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')

        plt.fill_between(alphas,
                scores + std_error,scores - std_error,alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])

        est_cv.alphas = alphas
        est_cv.random_state = 0
        k_fold = KFold(n_folds)

        scores = []
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            est_cv.fit(data.ix[train, 0:8].as_matrix(),
                data.ix[train, 8+self.cold: 9+self.cold].as_matrix())
            scores.append(est_cv.score(data.ix[test, 0:8].as_matrix(),
                data.ix[test, 8+self.cold: 9+self.cold].as_matrix()))

        plt.show()
        return np.mean(scores)*100
    
    return inner_wrap


class Ridge_M(Data, object):
    """
        Ridge regression model.
    """
    def __init__(self):
        Data.__init__(self)
    
    @staticmethod
    @cross_validator
    def fit():
        est = linear_model.Ridge()
        est_cv = linear_model.RidgeCV()
        return est, est_cv

    @property
    def accuracy(self):
        return Ridge_M.fit(self, self.data)


class ANN(Data, object):
    def __init__(self):
        Data.__init__(self)
        self.reg = MLPRegressor(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
        self.reg.fit(self.X_train,
                     self.y_train[:,self.cold:self.cold+1].ravel())

    @property
    def accuracy(self):
        return self.reg.score(self.X_test,
                        self.y_test[:,self.cold:self.cold+1].ravel())*100


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
    plt.bar(range(len(sp_result)), [x[0].correlation for x in sp_result],
            tick_label=data.label[:8], align="center")
    plt.title("Spearman correlation coefficient")
    plt.show()


def main():
    data = Data(cold=1)
    ridge = Ridge_M()
    ann = ANN()
    models = [("Ridge", ridge), ("ANN", ann)]
    
    for model_name, model_obj in models:
        print model_name, model_obj.accuracy
    sp_result = data.get_sp_rank()
    plot_feature_correlation(data)
    plot_model_accuracy(models)
    plot_pd(data)
    plot_spearman(sp_result, data)
    
    

if __name__ == "__main__":
    main()
