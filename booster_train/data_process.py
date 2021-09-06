import numpy as np
#data provider for REDD
def dataProvider(train1, train2, train3,windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    data_frame1 = pd.read_csv(train1,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame2 = pd.read_csv(train2,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame3 = pd.read_csv(train3,
                             #chunksize=10 ** 3,
                             header=0
                             )

    np_array = np.array(data_frame1)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features)
    labels0=np.array(labels)

    np_array = np.array(data_frame2)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features1=np.array(features)
    labels1=np.array(labels)

    np_array = np.array(data_frame3)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features2=np.array(features)
    labels2=np.array(labels)

    feature=np.concatenate((features0, features1), axis=0)
    feature=np.concatenate((feature, features2), axis=0)
    label=np.concatenate((labels0, labels1), axis=0)
    label=np.concatenate((label, labels2), axis=0)
    return feature, label

#data provider for UK-DALE
def dataProvider2(train1, train2, windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    data_frame1 = pd.read_csv(train1,
                             #chunksize=10 ** 3,
                             header=0
                             )
    data_frame2 = pd.read_csv(train2,
                             #chunksize=10 ** 3,
                             header=0
                             )

    np_array = np.array(data_frame1)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features0=np.array(features)
    labels0=np.array(labels)

    np_array = np.array(data_frame2)
    inputs, targets = np_array[:, 0], np_array[:, 1]
    window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    for i in range(0,window_num):
        inp=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(inp)
        labels.append(tar)
    features1=np.array(features)
    labels1=np.array(labels)
    feature=np.concatenate((features0, features1), axis=0)
    label=np.concatenate((labels0, labels1), axis=0)
    return feature, label

#data provider for Refit
def dataProvider3(path, windowsize):
    offset = int(0.5 * (windowsize - 1.0))
    feature_global = list()
    label_global = list()
    feature_global = np.array(feature_global).reshape(-1,windowsize)
    label_global = np.array(label_global)

    for idx, filename in enumerate(os.listdir(path)):
        data_frame=pd.read_csv(path+"/"+filename,
                             #chunksize=10 ** 3,
                             header=0
                             )
        np_array = np.array(data_frame)
        inputs, targets = np_array[:, 0], np_array[:, 1]
        window_num = inputs.size - 2 * offset
        features = list()
        labels = list()
        for i in range(0, window_num):
            inp = inputs[i:i + windowsize]
            tar = targets[i + offset]
            features.append(inp)
            labels.append(tar)
        features = np.array(features)
        print(features.shape)
        labels= np.array(labels)
        feature_global = np.concatenate((feature_global, features), axis=0)
        label_global = np.concatenate((label_global, labels), axis=0)
    return feature_global, label_global