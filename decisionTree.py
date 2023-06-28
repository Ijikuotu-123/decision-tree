import numpy as np
from util import get_data, get_dotnut,get_XOR
from datetime import datetime

def entropy(y):
    N = len(y)
    s1 = (y==1).sum()
    if 0 == s1 or N ==s1:
        return 0
    p1 = float(s1)/N
    p0 = 1 - p1
    return -p0*np.log2(p0) - p1*np.log2(p1)


class TreeNode:
    def __init__ (self, depth = 0, max_depth = None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X,Y):
        if len(Y) ==1 or len(set(Y)) == 1:      # base case
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
        else:
            D = X.shape[1] 
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X,Y,col) 
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split
            if max_ig == 0:             # base case. No more split to do
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = split

                if self.depth == self.max_depth:    # base case
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:, best_col] < split].mean()),
                        np.round(Y[X[:, best_col] >= split].mean()),
                    ]
                else:
                    left_id = (X[:,best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft,Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright,Yright)

    def find_split(self,X,Y,col):
        X_values = X[:,col]   # getting all the X for the given column
        sort_idx = np.argsort(X_values)  #this takes X_ values and arrrange in ascending order and returns the index
        X_values = X_values[sort_idx]
        Y_values = Y[sort_idx]

        boundaries = np.nonzero(Y_values[:,1] != Y_values[1:])[0] # getting indexes where y values changes
        best_split = None
        max_ig = 0
        for i in boundaries:
            split = (X_values[i] + X_values[i+1]) / 2
            ig = self.information_gain(X_values,Y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self,x,y,split):
        y0 = y[x < split]
        y1 = y[x >=split]
        N = len(y)
        y0len =len(y0)
        if y0len == 0 or y0len == 1:
            return 0
        p0 = float(len(y0))/N
        P1 = 1 -p0
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

    def predict_one(self,x):
        if self.col is not None and self.split is not None:  # checking for base cases. checking if a split has occured
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self,X):
        N =len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTree:
    def __init__(self, max_depth =None):
        self.max_depth = max_depth

    def fit(self, X,Y):
        self.root =TreeNode(max_depth=self.max_depth)
        self.root.fit(X,Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self,X,Y):
        P =self.predict(X)
        return np.mean(P==Y)


if __name__ == '__main__':
    X, Y = get_data()
    idx = np.logical_or(Y ==0 or Y == 1)  # helps to pick the indice where y = 0 or 1
    X = X[idx]
    Y = Y[idx]

    #X, Y = get_donut()
    #X, Y = get_xor()

    Ntrain = len(Y)/2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = DecisionTree()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("training time:", (datetime.now()- t0))

    t0 = datetime.now()
    print("train accuracy:", model.score(Xtrain,Ytrain))
    print("time to compute training accuracy:", (datetime.now()- t0))

    t0 = datetime.now()
    print("train accuracy:", model.score(Xtest,Ytest))
    print("time to compute test accuracy:", (datetime.now()- t0))















      




      






