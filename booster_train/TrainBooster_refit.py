from dask.distributed import Client, LocalCluster
from booster_train.FedBooster import *

cluster = LocalCluster(n_workers=*)
client = Client(cluster)

#add local runners into the pool; 3 runner as examples
worker0="IP:port"
worker1="IP:port"
worker2="IP:port"
worker3="IP:port"
worker4="IP:port"
workers=[worker0,worker1,worker2,worker3,worker4]

'''
Model training for selected targe appliances
--------------------------------------------
K valued at [10,20,40,60,80,99]
'''

trainfile1="/home/ubuntu/NILM/refit_training/dishwasher/dishwasher_house_9_training_.csv"
trainfile2="/home/ubuntu/NILM/refit_training/dishwasher/dishwasher_house_13_training_.csv"
trainfile3="/home/ubuntu/NILM/refit_training/dishwasher/dishwasher_house_16_training_.csv"
trainfile4="/home/ubuntu/NILM/refit_training/dishwasher/dishwasher_house_18_training_.csv"
trainfile5="/home/ubuntu/NILM/refit_training/dishwasher/dishwasher_house_20_training_.csv"
fileList=[trainfile1, trainfile2, trainfile3,trainfile4, trainfile5]
for k in [10,20,40,60,80,99]:
    start=time.time()
    print("start time is:", start)
    booster=DaskBooster(k, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.02145, scoring="mae",  maxbins=50, maxdepth=10)
    booster.fit(fileList)
    print("The time cost is:",time.time()-start)
    booster.client=None
    model0='/home/ubuntu/NILM/refit_training/converK/dishwasher1_'+str(k)+'.pkl'
    with open(model0, 'wb+') as f:
        boost= pickle.dump(booster, f)

trainfile1="/home/ubuntu/NILM/refit_training/fridge/fridge_house_2_training_.csv"
trainfile2="/home/ubuntu/NILM/refit_training/fridge/fridge_house_5_training_.csv"
trainfile3="/home/ubuntu/NILM/refit_training/fridge/fridge_house_9_training_.csv"
trainfile4="/home/ubuntu/NILM/refit_training/fridge/fridge_house_12_training_.csv"
trainfile5="/home/ubuntu/NILM/refit_training/fridge/fridge_house_15_training_.csv"
fileList=[trainfile1, trainfile2, trainfile3,trainfile4, trainfile5]
for k in [10,20,40,60,80,99]:
    start=time.time()
    print("start time is:", start)
    booster=DaskBooster(k, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.02145, scoring="mae",  maxbins=50, maxdepth=10)
    booster.fit(fileList)
    print("The time cost is:",time.time()-start)
    booster.client=None
    model0='/home/ubuntu/NILM/refit_training/converK/fridge_'+str(k)+'.pkl'
    with open(model0, 'wb+') as f:
        boost= pickle.dump(booster, f)

trainfile1="/home/ubuntu/NILM/refit_training/microwave/microwave_house_4_training_.csv"
trainfile2="/home/ubuntu/NILM/refit_training/microwave/microwave_house_10_training_.csv"
trainfile3="/home/ubuntu/NILM/refit_training/microwave/microwave_house_12_training_.csv"
trainfile4="/home/ubuntu/NILM/refit_training/microwave/microwave_house_17_training_.csv"
trainfile5="/home/ubuntu/NILM/refit_training/microwave/microwave_house_19_training_.csv"
fileList=[trainfile1, trainfile2, trainfile3,trainfile4, trainfile5]
for k in [10,20,40,60,80,99]:
    start=time.time()
    print("start time is:", start)
    booster=DaskBooster(k, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.02145, scoring="mae",  maxbins=50, maxdepth=10)
    booster.fit(fileList)
    print("The time cost is:",time.time()-start)
    booster.client=None
    model0='/home/ubuntu/NILM/refit_training/converK/microwave_'+str(k)+'.pkl'
    with open(model0, 'wb+') as f:
        boost= pickle.dump(booster, f)

trainfile1="/home/ubuntu/NILM/refit_training/washingmachine/washingmachine_house_9_training_.csv"
trainfile2="/home/ubuntu/NILM/refit_training/washingmachine/washingmachine_house_15_training_.csv"
trainfile3="/home/ubuntu/NILM/refit_training/washingmachine/washingmachine_house_16_training_.csv"
trainfile4="/home/ubuntu/NILM/refit_training/washingmachine/washingmachine_house_17_training_.csv"
trainfile5="/home/ubuntu/NILM/refit_training/washingmachine/washingmachine_house_18_training_.csv"
fileList=[trainfile1, trainfile2, trainfile3,trainfile4, trainfile5]
for k in [10,20,40,60,80,99]:
    start=time.time()
    print("start time is:", start)
    booster=DaskBooster(k, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.02145, scoring="mae",  maxbins=50, maxdepth=10)
    booster.fit(fileList)
    print("The time cost is:",time.time()-start)
    booster.client=None
    model0='/home/ubuntu/NILM/refit_training/converK/washingmachine_'+str(k)+'.pkl'
    with open(model0, 'wb+') as f:
        boost= pickle.dump(booster, f)

trainfile1="/home/ubuntu/NILM/refit_training/kettle/kettle_house_2_training_.csv"
trainfile2="/home/ubuntu/NILM/refit_training/kettle/kettle_house_3_training_.csv"
trainfile3="/home/ubuntu/NILM/refit_training/kettle/kettle_house_4_training_.csv"
trainfile4="/home/ubuntu/NILM/refit_training/kettle/kettle_house_5_training_.csv"
trainfile5="/home/ubuntu/NILM/refit_training/kettle/kettle_house_6_training_.csv"
fileList=[trainfile1, trainfile2, trainfile3,trainfile4, trainfile5]
for k in [10,20,40,60,80,99]:
    start=time.time()
    print("start time is:", start)
    booster=DaskBooster(k, workers, client=client, num_round=100, eta=0.23179, gamma=0, Lambda=0.02145, scoring="mae",  maxbins=50, maxdepth=10)
    booster.fit(fileList)
    print("The time cost is:",time.time()-start)
    booster.client=None
    model0='/home/ubuntu/NILM/refit_training/converK/kettle1_'+str(k)+'.pkl'
    with open(model0, 'wb+') as f:
        boost= pickle.dump(booster, f)

