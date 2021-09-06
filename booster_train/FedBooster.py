import numpy as np
import os
import pandas as pd
import sklearn
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from math import sqrt
import re
import math
import pickle
from sklearn.metrics import  zero_one_loss,mean_absolute_error,r2_score
from random import sample
from dask.distributed import wait
from graphviz import Digraph

print(sklearn.__version__) #check sklearn version (update 0.24.2)


def CalGain(listL, listR, firstG, secondG, Lambda, gamma):
    """calculater Gain for histogram"""
    GL = firstG[listL].sum()
    GR = firstG[listR].sum()
    HL = secondG[listL].sum()
    HR = secondG[listR].sum()
    return (GL ** 2 / (HL + Lambda) + GR ** 2 / (HR + Lambda) - (GR + GL) ** 2 / (
                HL + HR + Lambda)) / 2 - gamma

def getListMaxNumIndex(num_list,topk=10):
    """Select topK candidates' indexes"""
    num_dict={}
    for i in range(len(num_list)):
        num_dict[i]=num_list[i]
    res_list=sorted(num_dict.items(),key=lambda e:e[1])
    max_num_index=[one[0] for one in res_list[::-1][:topk]]
    return max_num_index


#####################all method/task executed by workers##################################################
def getPara(future,i):
    return future[i]

#initialise prediction values
def inithaty(dataSample):
    haty = np.zeros(len(dataSample))
    return haty
#initialise the value of second-order gradients
def inith(dataSample):
    h = np.ones(len(dataSample)) * 2
    return h
#calculate first-order gradients of samples located at any certain worker
def calG(y_train, haty):
    return -2 * (y_train - haty)

def initf(x_train):
    return np.empty(len(x_train))

def initindex(x_train):
    return np.arange(x_train.shape[0])

#calculate f of current tree
def assignf(indexlist, f, w):
    f[indexlist]=w
    return True

def newNode(x_train, y_train, indexlist, bestSplitFeature, bestSplitValue):
    left_index = x_train[:, bestSplitFeature] <= bestSplitValue
    sub_X_train_left, sub_y_train_left = x_train[left_index], y_train[left_index]
    sub_X_train_right, sub_y_train_right = x_train[~left_index], y_train[~left_index]
    indexlist_left = indexlist[left_index]
    indexlist_right = indexlist[~left_index]
    return sub_X_train_left, sub_y_train_left, sub_X_train_right, sub_y_train_right, indexlist_left, indexlist_right

#update the value of prediction output at current stage
def updateHaty(haty, f, eta):
    updateHaty=haty + eta * f
    return updateHaty


def getLoss_worker(y_train, haty):
    return mean_squared_error(y_train, haty, squared=False)


def find_split_worker(x_train, g, h, indexlist, k, bins, Lambda, gamma):
    topKsplit = []
    topKgain = []
    firstG = g[indexlist]
    secondG = h[indexlist]
    for feature in range(x_train.shape[1]):
        bestGain = 0
        bestSplitValue = -1
        AllValue = sorted(set(x_train[:, feature]))
        if len(AllValue) > bins:
            try:
                ValueSet = sample(AllValue, bins)
            except:
                print("length of Allvalue is :", len(AllValue))
                print("length of self.bins is :", bins)
        else:
            ValueSet = AllValue
        for Val in ValueSet:
            boolindexLeft = x_train[:, feature] <= Val
            boolindexRight = ~boolindexLeft
            indexLeft = boolindexLeft
            indexRight = boolindexRight

            gain = CalGain(indexLeft, indexRight, firstG, secondG, Lambda, gamma)
            if gain > bestGain:
                bestGain = gain
                bestSplitValue = Val
        topKsplit.append({feature: bestSplitValue})
        topKgain.append(bestGain)
    index = getListMaxNumIndex(topKgain, topk=k)
    topgainList = []
    topsplitList = []
    for i in index:
        topgainList.append(topKgain[i])
        topsplitList.append(topKsplit[i])
    return topgainList, topsplitList


def seq2point(dataframe):
    np_array = np.array(dataframe)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    t = np.array([i for i in range(len(inputs))])
    windowsize = 19
    offset = int(0.5 * (windowsize - 1.0))
    window_num = inputs.size - 2 * offset
    features = list()
    labels = list()
    for i in range(0, window_num):
        data_in = inputs[i:i + windowsize]
        tar = targets[i + offset]
        features.append(data_in)
        labels.append(tar)
    X = np.array(features)
    Y = np.array(labels)
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.2, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    return x_train_all, y_train_all, x_predict, y_predict, x_test, y_test


class DaskBooster():
    def __init__(self, k, workers, client=None, num_round=50, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mae",
                 maxbins=500, maxdepth=10):
        self.client = client
        self.scoring = scoring
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
        self.workers = workers
        self.bins = maxbins
        self.trace = []
        self.k = k
        self.depth = maxdepth
        self.worker_trace = []

    def getLoss(self, y_train):
        haty = self.client.gather(self.haty)
        y = self.client.gather(y_train)
        haty = np.concatenate(haty, axis=0)
        y = np.concatenate(y, axis=0)
        return mean_squared_error(y, haty, squared=False), mean_absolute_error(y, haty)

    #calculate g_i
    def _G(self, y_train, haty):

        return -2 * (y_train - haty)

    #calculater Gain for histogram
    def _Gain(self, listL, listR, firstG, secondG):
        GL = firstG[listL].sum()
        GR = firstG[listR].sum()
        HL = secondG[listL].sum()
        HR = secondG[listR].sum()

        return (GL ** 2 / (HL + self.Lambda) + GR ** 2 / (HR + self.Lambda) - (GR + GL) ** 2 / (
                HL + HR + self.Lambda)) / 2 - self.gamma

    def _w(self, indexlist):
        first = self.client.gather(self.firstG)
        second = self.client.gather(self.secondG)
        indexlist = self.client.gather(indexlist)
        sumG = 0
        sumH = 0
        for i in range(len(first)):
            sumG += np.sum(first[i][indexlist[i]])
            sumH += np.sum(second[i][indexlist[i]])
        return -sumG / (sumH + self.Lambda)

    #Find the best split point
    def BestSplit(self, x_train, indexlist):
        bestGain = 0
        bestSplitFeature = -1
        bestSplitValue = -1
        x_train_worker = x_train
        x_all_set = self.client.gather(x_train)
        x_train = np.concatenate(x_all_set, axis=0)
        firstG = []
        secondG = []
        indexlist = self.client.gather(indexlist)
        g = self.client.gather(self.firstG)
        for i in range(len(g)):
            firstG.append(g[i][indexlist[i]])
        firstG = np.concatenate(firstG, axis=0)
        h = self.client.gather(self.secondG)

        for i in range(len(h)):
            secondG.append(h[i][indexlist[i]])
        secondG = np.concatenate(secondG, axis=0)

        splitCandidates = []
        for i in range(len(indexlist)):
            splitCandidates.append(
                self.client.submit(find_split_worker, x_train_worker[i], self.firstG[i], self.secondG[i], indexlist[i],
                                   self.k, self.bins, self.Lambda, self.gamma, workers=self.workers[i]))
        wait(splitCandidates)
        splitCandidates = self.client.gather(splitCandidates)
        candidates = list()
        "Derive topK candidates from feature list"
        for element in splitCandidates:
            for feature in element[1]:
                candidates.append(list(feature.keys())[0])
        candidates = np.array(candidates)
        candiset, cnt = np.unique(candidates, return_counts=True)
        cnt = cnt.tolist()
        max_num_index = getListMaxNumIndex(cnt, topk=self.k)

        for feature in candiset[max_num_index]:
            # print(feature)
            AllValue = sorted(set(x_train[:, feature]))
            if len(AllValue) > self.bins:
                try:
                    ValueSet = sample(AllValue, self.bins)
                except:
                    print("length of Allvalue is :", len(AllValue))
                    print("length of self.bins is :", self.bins)
            else:
                ValueSet = AllValue
            for Val in ValueSet:
                boolindexLeft = x_train[:, feature] <= Val
                boolindexRight = ~boolindexLeft
                indexLeft = boolindexLeft
                indexRight = boolindexRight
                gain = self._Gain(indexLeft, indexRight, firstG, secondG)

                if gain > bestGain:
                    bestGain = gain
                    bestSplitFeature = feature
                    bestSplitValue = Val
        if bestSplitFeature == -1:
            return None, None
        else:
            return bestSplitFeature, bestSplitValue

    def create_tree(self, x_train, y_train, depth, indexlists):
            bestSplitFeature, bestSplitValue = self.BestSplit(x_train, indexlists)
            numSample=0
            for indexlist in indexlists:
                numSample+=indexlist.result().shape[0]
            if (bestSplitFeature is None or numSample<200 or depth>self.depth):
                w = self._w(indexlists)
                allweight=[]
                for i in range(len(indexlists)):
                    curf=self.client.submit(assignf, indexlists[i], self.f[i], w, workers=self.workers[i])
                    allweight.append(curf)
                wait(allweight)
                return w
            else:
                depth+=1
                sub_X_train_left, sub_y_train_left, sub_X_train_right, sub_y_train_right, indexlist_left, indexlist_right=[],[],[],[],[],[]
                childPara=[]
                for i in range(len(indexlists)):
                    childPara.append(self.client.submit(newNode, x_train[i], y_train[i], indexlists[i], bestSplitFeature, bestSplitValue, workers=self.workers[i]))
                wait(childPara)
                i=0
                for child in childPara:
                    X_train_left=self.client.submit(getPara, child, 0, workers=self.workers[i])
                    y_train_left=self.client.submit(getPara, child, 1, workers=self.workers[i])
                    X_train_right=self.client.submit(getPara, child, 2, workers=self.workers[i])
                    y_train_right=self.client.submit(getPara, child, 3, workers=self.workers[i])
                    leftindex=self.client.submit(getPara, child, 4, workers=self.workers[i])
                    rightindex=self.client.submit(getPara, child, 5, workers=self.workers[i])
                    sub_X_train_left.append(X_train_left)
                    sub_y_train_left.append(y_train_left)
                    sub_X_train_right.append(X_train_right)
                    sub_y_train_right.append(y_train_right)
                    indexlist_left.append(leftindex)
                    indexlist_right.append(rightindex)
                    i+=1
                wait(sub_X_train_left)
                wait(sub_y_train_left)
                wait(sub_X_train_right)
                wait(sub_y_train_right)
                wait(indexlist_left)
                wait(indexlist_right)

                leftchild = self.create_tree(sub_X_train_left, sub_y_train_left, depth, indexlist_left)
                rightchild = self.create_tree(sub_X_train_right, sub_y_train_right, depth, indexlist_right)
                return {bestSplitFeature: {"<={}".format(bestSplitValue): leftchild,
                                            ">{}".format(bestSplitValue): rightchild}}

    def fit(self, fileList):
        data_frames = []
        for i in range(len(fileList)):
            data_frame = pd.read_csv(fileList[i], header=0)
            data_frames.append(data_frame)

        allData = self.client.map(seq2point, data_frames, workers=self.workers)
        wait(allData)
        x_train = self.client.map(getPara, allData, [0, 0, 0], workers=self.workers)
        y_train = self.client.map(getPara, allData, [1, 1, 1], workers=self.workers)
        wait(x_train)
        wait(y_train)
        self.haty = []
        self.secondG = []
        self.indexLists = []

        for i in range(len(fileList)):
            weight = self.client.submit(inithaty, x_train[i], workers=self.workers[i])
            self.haty.append(weight)
        wait(self.haty)

        for i in range(len(fileList)):
            hess = self.client.submit(inith, x_train[i], workers=self.workers[i])
            self.secondG.append(hess)
        wait(self.secondG)

        for i in range(len(fileList)):
            indexlist = self.client.submit(initindex, x_train[i], workers=self.workers[i])
            self.indexLists.append(indexlist)
        wait(self.indexLists)


        for index in range(self.num_round):
            self.firstG = []
            self.f = []
            print('start the tree0: ', index)
            for i in range(len(fileList)):
                grad = self.client.submit(calG, y_train[i], self.haty[i], workers=self.workers[i])
                self.firstG.append(grad)
            wait(self.firstG)

            for i in range(len(fileList)):
                currentf = self.client.submit(initf, x_train[i], workers=self.workers[i])
                self.f.append(currentf)
            wait(self.f)
            if index == 0:
                loss, mae = self.getLoss(y_train)
                self.trace.append(loss)
                print("initail mae is:", mae)
                print("initail loss is:", loss)
                currentLoss = []
                for i in range(len(fileList)):
                    currentLoss.append(
                        self.client.submit(getLoss_worker, y_train[i], self.haty[i], workers=self.workers[i]))
                wait(currentLoss)
                self.worker_trace.append(self.client.gather(currentLoss))

            newtree = self.create_tree(x_train, y_train, 0, self.indexLists)
            newhaty = []
            for i in range(len(fileList)):
                a = self.client.submit(updateHaty, self.haty[i], self.f[i], self.eta, workers=self.workers[i])
                newhaty.append(a)
            wait(newhaty)
            self.haty = newhaty

            currentLoss = []
            for i in range(len(fileList)):
                currentLoss.append(
                    self.client.submit(getLoss_worker, y_train[i], self.haty[i], workers=self.workers[i]))
            wait(currentLoss)
            self.worker_trace.append(self.client.gather(currentLoss))
            loss, mae = self.getLoss(y_train)
            print("converage loss is:", loss)
            print("mae in this step is", mae)
            self.trace.append(loss)
            self.ensemble.append(newtree)
        # f = open("/NILM/test/modelt1.txt", 'w')
        # f.write(str(self.ensemble))
        # f.close()
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











