import numpy as np
from random import sample
from sklearn.metrics import mean_squared_error, zero_one_loss
from booster_train.data_process import *
from graphviz import Digraph
from sklearn.model_selection import train_test_split

def getLoss_worker(y_train, haty):
      return mean_squared_error(y_train, haty,squared=False)
class OriginBooster():
    def __init__(self, num_round=50, eta=0.23179, gamma=0, Lambda=0.7145, maxbins=500, maxdepth=10):
        # self.scoring = scoring
        self.num_round = num_round
        self.eta = eta
        self.gamma = gamma
        self.Lambda = Lambda
        self.ensemble = []
        self.firstG = None  # first order gradient
        self.secondG = None  # second order gradient
        self.haty = None
        self.f = None
        self.indexList = None
        self.bins = 500
        self.trace = []
        self.depth = maxdepth

    def _G(self, y_train):
        return -2 * (y_train - self.haty)

    def _Gain(self, listL, listR):
        GL = self.g[listL].sum()
        GR = self.g[listR].sum()
        HL = self.h[listL].sum()
        HR = self.h[listR].sum()

        return (GL ** 2 / (HL + self.Lambda) + GR ** 2 / (HR + self.Lambda) - (GR + GL) ** 2 / (
                HL + HR + self.Lambda)) / 2 - self.gamma

    def _w(self, indexlist):
        return -np.sum(self.g[indexlist]) / (np.sum(self.h[indexlist]) + self.Lambda)

    def BestSplit(self, X_train, indexlist):
        bestGain = 0
        bestSplitFeature = -1
        bestSplitValue = -1
        for feature in range(X_train.shape[1]):
            AllValue = sorted(set(X_train[:, feature]))
            if len(AllValue) > self.bins:
                try:
                    ValueSet = sample(AllValue, self.bins)
                except:
                    print("length of Allvalue is :", len(AllValue))
                    print("length of self.bins is :", self.bins)
            else:
                ValueSet = AllValue
            for Val in ValueSet:
                boolindexLeft = X_train[:, feature] <= Val
                boolindexRight = ~boolindexLeft
                indexLeft = indexlist[boolindexLeft]
                indexRight = indexlist[boolindexRight]
                gain = self._Gain(indexLeft, indexRight)
                if gain > bestGain:
                    bestGain = gain
                    bestSplitFeature = feature
                    bestSplitValue = Val
        if bestSplitFeature == -1 or bestGain == 0:
            return None, None
        else:
            return bestSplitFeature, bestSplitValue

    def create_tree(self, X_train, y_train, depth, indexlist):
        bestSplitFeature, bestSplitValue = self.BestSplit(X_train, indexlist)
        if (bestSplitFeature is None or len(indexlist) < 200 or depth >= self.depth):
            w = self._w(indexlist)
            self.f[indexlist] = w
            return w
        else:
            depth += 1
            left_index = X_train[:, bestSplitFeature] <= bestSplitValue
            sub_X_train_left, sub_y_train_left = X_train[left_index], y_train[left_index]
            sub_X_train_right, sub_y_train_right = X_train[~left_index], y_train[~left_index]
            indexlist_left = indexlist[left_index]
            indexlist_right = indexlist[~left_index]
            leftchild = self.create_tree(sub_X_train_left, sub_y_train_left, depth, indexlist=indexlist_left)
            rightchild = self.create_tree(sub_X_train_right, sub_y_train_right, depth, indexlist=indexlist_right)
            return {bestSplitFeature: {"<={}".format(bestSplitValue): leftchild,
                                       ">{}".format(bestSplitValue): rightchild}}

    def fit(self, X_train, y_train):
        self.haty = np.zeros(len(X_train))
        self.h = np.ones(len(X_train)) * 2

        for index in range(self.num_round):
            self.g = self._G(y_train)
            self.f = np.empty(len(X_train))
            if index == 0:
                loss = getLoss_worker(y_train, self.haty)
                print("initial loss is:", loss)
                self.trace.append(loss)
            newtree = self.create_tree(X_train, y_train, 0, np.arange(len(X_train)))
            print('have finished one tree: ', index)
            self.ensemble.append(newtree)
            self.haty = self.haty + self.eta * self.f
            loss = getLoss_worker(y_train, self.haty)
            print("loss is :", loss)
            self.trace.append(loss)
        return

    def draw_one_tree(self, index):
        def export_graphviz(tree, root_index):
            root = next(iter(tree))
            text_node.append([str(root_index), "feature:{}".format(root)])
            secondDic = tree[root]
            for key in secondDic:
                if type(secondDic[key]) == dict:
                    i[0] += 1
                    secondrootindex = i[0]
                    text_edge.append([str(root_index), str(secondrootindex), str(key)])
                    export_graphviz(secondDic[key], secondrootindex)
                else:
                    i[0] += 1
                    text_node.append([str(i[0]), str(secondDic[key])])
                    text_edge.append([str(root_index), str(i[0]), str(key)])

        tree = self.ensemble[index]
        text_node = []
        text_edge = []
        i = [1]
        export_graphviz(tree, i[0])
        dot = Digraph()
        for line in text_node:
            dot.node(line[0], line[1])
        for line in text_edge:
            dot.edge(line[0], line[1], line[2])

        dot.view()

    def predict(self, X_test):
        return np.array([self._predict(test) for test in X_test])

    def _predict(self, test):
        def __predict(tree, test):
            feature = next(iter(tree))
            secondDic = tree[feature]
            content = test[feature]
            for key in secondDic:
                if eval(str(content) + key):
                    if type(secondDic[key]) == dict:
                        return __predict(secondDic[key], test)
                    else:
                        return secondDic[key]

        assert len(self.ensemble) != 0, "fit before predict"
        res = 0
        for i in range(len(self.ensemble)):
            tree = self.ensemble[i]
            res_temp = __predict(tree, test)
            res += res_temp * self.eta
        return res

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        if self.scoring == "mse":
            return sum((y_test - y_pre) ** 2) / len(X_test)
        elif self.scoring == "r2":
            return 1 - sum((y_test - y_pre) ** 2) / sum((y_test - y_test.mean()) ** 2)

    def get_params(self, deep=False):
        dic = {}
        dic["num_round"] = self.num_round
        dic["eta"] = self.eta
        dic["gamma"] = self.gamma
        dic["Lambda"] = self.Lambda
        dic["scoring"] = self.scoring
        return dic
''' Example for running OriginBooster'''
path="#refit_path"
X,Y=dataProvider3(path ,99)
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.01, random_state=100)
booster=OriginBooster(num_round=100, eta=0.23179, gamma=0.00001, Lambda=0.02145,  maxbins=500, maxdepth=10)
booster.fit(x_train_all,y_train_all)
print(booster)