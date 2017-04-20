# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class Data(object):
    """
        Class for pre-processing data.
    """
    def __init__(self, filename="data.csv", cold=0):
        self.data = pd.read_csv(filename)
        self.cold = cold
        X, y = self.data.ix[:,0:8].as_matrix(), self.data.ix[:,8:10].as_matrix()
        scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y,
                                                                                test_size=0.2,
                                                                                random_state=0)
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __repr__(self):
        pass


class Linear(Data, object):
    """
        Linear regression model.
    """
    def __init__(self):
        Data.__init__(self)
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.X_train, self.y_train[:,self.cold:self.cold+1].ravel())

    @property
    def params(self):
        return self.reg.coef_

    @property
    def accuracy(self):
        return self.reg.score(self.X_test, self.y_test[:,self.cold:self.cold+1].ravel())*100


class Ridge(Data, object):
    """
        Ridge regression model.
    """
    def __init__(self):
        Data.__init__(self)
        self.reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.reg.fit(self.X_train, self.y_train[:,self.cold:self.cold+1].ravel())

    @property
    def params(self):
        return self.reg.coef_, self.reg.alpha_

    @property
    def accuracy(self):
        return self.reg.score(self.X_test, self.y_test[:,self.cold:self.cold+1].ravel())*100


class ANN(Data, object):
    def __init__(self):
        Data.__init__(self)
        self.reg = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=(5, 2), random_state=1)
        self.reg.fit(self.X_train, self.y_train[:,self.cold:self.cold+1].ravel())
        

    @property
    def params(self):
        return self.reg.coef_

    @property
    def accuracy(self):
        return self.reg.score(self.X_test, self.y_test[:,self.cold:self.cold+1].ravel())*100


class SVRM(Data, object):
    def __init__(self):
        Data.__init__(self)
        self.reg = svm.SVR(kernel='rbf')
        self.reg.fit(self.X_train, self.y_train[:,self.cold:self.cold+1].ravel())

    @property
    def params(self):
        return self.reg.coef_

    @property
    def accuracy(self):
        return self.reg.score(self.X_test, self.y_test[:,self.cold:self.cold+1].ravel())*100


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
    plt.ylim(90,100)
    plt.show()

def plot_feature_correlation(data):
    """
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
    label = ["Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
             "Overall Height", "Orientation", "Glazing Area",
             "Glazing Area Distribution", "Heating Load", "Cooling Load"]
    cycol = cycle('bgrcmk').next
    nf = len(label)-2    
    
    for i in range(nf):
        for j in range(nf):
            plt.subplot(nf, nf, i*nf+j+1)
            plt.axis('off')
            plt.scatter(data.data.ix[:, j:j+1].as_matrix(),
                        data.data.ix[:, i:i+1].as_matrix(),
                        color = cycol())        
    
    plt.show()


def main():
    data = Data(cold=1)
    linear = Linear()
    ridge = Ridge()
    ann = ANN()
    svr = SVRM()
    models = [("Linear", linear), ("Ridge", ridge),
              ("ANN", ann), ("SVM", svr)]
    for model_name, model_obj in models: print model_name, model_obj.accuracy
    plot_feature_correlation(data)
    plot_model_accuracy(models)
    

if __name__ == "__main__":
    main()
                                                                                
        