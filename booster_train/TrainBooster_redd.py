from dask.distributed import Client, LocalCluster
from booster_train.FedBooster import *

cluster = LocalCluster(n_workers=*)
client = Client(cluster)

#add local runners into the pool; 3 runner as examples
worker0="IP:port"
worker1="IP:port"
worker2="IP:port"
workers=[worker0,worker1,worker2]

# model training for dishwasher
booster0=DaskBooster(9, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mae",  maxbins=500, maxdepth=10)
trainfile1 = "/home/ubuntu/NILM/training_data/dishwasher_house_2_training_.csv"
trainfile2 = "/home/ubuntu/NILM/training_data/dishwasher_house_3_training_.csv"
trainfile0="/home/ubuntu/NILM/training_data/dishwasher_test_.csv"
fileList=[trainfile0, trainfile1, trainfile2]
booster0.fit(fileList)
model0 = '/home/ubuntu/NILM/redd_training/dishwasher_.pkl'
with open(model0, 'wb+') as f:
    boost = pickle.dump(booster0, f)

booster1=DaskBooster(9, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mae",  maxbins=500, maxdepth=10)
trainfile1 = "/home/ubuntu/NILM/training_data/fridge_house_2_training_.csv"
trainfile2 = "/home/ubuntu/NILM/training_data/fridge_house_3_training_.csv"
trainfile0="/home/ubuntu/NILM/training_data/fridge_test_.csv"
fileList=[trainfile0, trainfile1, trainfile2]
booster1.fit(fileList)
model1 = '/home/ubuntu/NILM/redd_training/fridge_.pkl'
with open(model1, 'wb+') as f:
    boost = pickle.dump(booster1, f)

booster2=DaskBooster(9, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mae",  maxbins=500, maxdepth=10)
trainfile1 = "/home/ubuntu/NILM/training_data/washingmachine_house_2_training_.csv"
trainfile2 = "/home/ubuntu/NILM/training_data/washingmachine_house_3_training_.csv"
trainfile0="/home/ubuntu/NILM/training_data/washingmachine_test_.csv"
fileList=[trainfile0, trainfile1, trainfile2]
booster2.fit(fileList)
model2 = '/home/ubuntu/NILM/redd_training/washingmachine_.pkl'
with open(model2, 'wb+') as f:
    boost = pickle.dump(booster2, f)

booster3=DaskBooster(9, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.7145, scoring="mae",  maxbins=500, maxdepth=10)
trainfile1 = "/home/ubuntu/NILM/training_data/microwave_house_2_training_.csv"
trainfile2 = "/home/ubuntu/NILM/training_data/microwave_house_3_training_.csv"
trainfile0="/home/ubuntu/NILM/training_data/microwave_test_.csv"
fileList=[trainfile0, trainfile1, trainfile2]
booster3.fit(fileList)
model3 = '/home/ubuntu/NILM/redd_training/microwave_.pkl'
with open(model3, 'wb+') as f:
    boost = pickle.dump(booster3, f)