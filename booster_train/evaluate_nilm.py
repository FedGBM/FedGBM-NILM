from math import sqrt
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
'''
evaluate the disaggregation deviation

Parameters:
----------------
target: the groud truth , np.array
prediction: the prediction, np.array
threshold:float -on/off state detection  (refer to appliance parameters on NILM_data_management)
'''



def get_mae(target, prediction):
    assert (target.shape == prediction.shape)

    return mean_absolute_error(target, prediction)

def get_sae(target, prediction):
    assert (target.shape == prediction.shape)

    r = target.sum()
    r0 = prediction.sum()
    sae = abs(r0 - r) / r
    return sae

def get_nde(target, prediction):
    assert (target.shape == prediction.shape)

    error, squarey = [], []
    for i in range(len(prediction)):
        value = prediction[i] - target[i]
        error.append(value * value)
        squarey.append(target[i] * target[i])
    nde = sqrt(sum(error) / sum(squarey))
    return nde



def get_TP(target, prediction, threshold):
    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold
    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)
    return tp


def get_FP(target, prediction, threshold):

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold
    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)
    return fp


def get_FN(target, prediction, threshold):

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold
    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)
    return fn


def get_TN(target, prediction, threshold):

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold
    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)
    return tn


def get_recall(target, prediction, threshold):
    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    log('tp={0}'.format(tp))
    log('fn={0}'.format(fn))
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):

    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    log('tp={0}'.format(tp))
    log('fp={0}'.format(fp))
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):

    recall = get_recall(target, prediction, threshold)
    precision = get_precision(target, prediction, threshold)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):

    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)
    accuracy = (tp + tn) / target.size
    return accuracy

'''
evaluate the disaggregation deviation

Parameters:
----------------
timelist: training costs across different K values(number of candidate features)
app: accuracy of different models
'''
def score(timelist, app):
    featurenum=[10,20,40,60,80,99]
    timelist=np.array(timelist).reshape(-1,1)
    app = np.array(app).reshape(-1, 1)
    featurenum = np.array(featurenum).reshape(-1, 1)
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler3= MinMaxScaler()
    scaler1.fit(timelist)
    scaler2.fit(app)
    scaler3.fit(featurenum)
    timelist=scaler1.transform(timelist)
    app= scaler2.transform(app)
    featurenum = scaler3.transform(featurenum)
    score=[]
    for i in range(len(timelist)-1):
        traincost=(timelist[-1][0]-timelist[i][0])
        privacygain =  featurenum[i][0]
        accuracy=(app[i][0]-app[-1][0])
        score.append(traincost/(accuracy+privacygain))
    return score

def get_statistic_property(data):

    mean = np.mean(data)
    std = np.std(data)
    min_v = np.sort(data)[0]
    max_v = np.sort(data)[-1]

    quartile1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    quartile2 = np.percentile(data, 75)

    return mean, std, min_v, max_v, quartile1, median, quartile2