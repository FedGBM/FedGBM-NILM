import pandas as pd
import time
import os
import re
import argparse

DATA_DIRECTORY = 'path/NILM_dataset/data/ALLDATA/REFIT/'
SAVE_PATH = 'path/NILM/refit_training/microwave/'
AGG_MEAN = 522
AGG_STD = 814


def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--appliance_name', type=str, default='microwave',
                        help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean', type=int, default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std', type=int, default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='The directory to store the training data')
    return parser.parse_args()


params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        'houses': [4, 10, 12, 17, 19],
        'channels': [8, 8, 3, 7, 4],
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        'houses': [2, 5, 9, 12, 15],
        'channels': [1, 1, 1, 1, 1],
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        'houses': [5, 7, 9, 13, 16, 18, 20],
        'channels': [4, 6, 4, 4, 6, 6, 5],
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
    }
}


def load(path, building, appliance, channel):
    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel + 2],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv


def main():
    start_time = time.time()

    args = get_arguments()

    appliance_name = args.appliance_name
    print(appliance_name)

    path = args.data_dir
    save_path = args.save_path
    if not os.path.exists(appliance_name):
        os.makedirs(appliance_name)
    save_path = appliance_name + '/'
    print(path)


    total_length = 0
    print("Starting creating dataset...")
    for idx, filename in enumerate(os.listdir(path)):
        single_step_time = time.time()
        if int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['houses']:
            print('File: ' + filename)
            print('    House: ' + re.search(r'\d+', filename).group())
            print(type(re.search(r'\d+', filename).group()))
            # Loading
            try:
                csv = load(path,
                           int(re.search(r'\d+', filename).group()),
                           appliance_name,
                           params_appliance[appliance_name]['channels']
                           [params_appliance[appliance_name]['houses']
                           .index(int(re.search(r'\d+', filename).group()))]
                           )
                rows, columns = csv.shape
                total_length += rows
                # saving the whole merged file
                csv.to_csv(args.save_path + appliance_name + '_house_' + re.search(r'\d+',
                                                                                   filename).group() + '_training_.csv',
                           mode='a', index=False, header=False)
                del csv
            except:
                pass
        else:
            continue


if __name__ == '__main__':
    main()
