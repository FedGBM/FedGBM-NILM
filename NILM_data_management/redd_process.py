import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os


params_appliance = {
    'microwave': {
        'windowlength': 99,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        'houses': [1, 2, 3],
        'channels': [11, 6, 16],
    },
    'fridge': {
        'windowlength': 99,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        'houses': [1, 2, 3],
        'channels': [5, 9, 7],
    },
    'dishwasher': {
        'windowlength': 99,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        'houses': [1, 2, 3],
        'channels': [6, 10, 9],
    },
    'washingmachine': {
        'windowlength': 99,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        'houses': [1, 2, 3],
        'channels': [20, 7, 13],
    }
}


DATA_DIRECTORY = 'path/NILM_dataset/low_freq'
SAVE_PATH = 'path/NILM/training_data/'
AGG_MEAN = 522
AGG_STD = 814
def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the REDD data')
    parser.add_argument('--appliance_name', type=str, default='washingmachine',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()


start_time = time.time()
args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)


def main():

    sample_seconds = 8
    nrows = None
    debug = False

    appliance_name = args.appliance_name
    print('\n' + appliance_name)
    train = pd.DataFrame(columns=['aggregate', appliance_name])

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + args.data_dir + 'house_' + str(h) + '/'
                + 'channel_' +
                str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
                '.dat')

        # read data
        mains1_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                      str(1) + '.dat',
                                      sep="\s+",
                                      nrows=nrows,
                                      usecols=[0, 1],
                                      names=['time', 'mains1'],
                                      dtype={'time': str},
                                      )

        mains2_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                      str(2) + '.dat',
                                      sep="\s+",
                                      nrows=nrows,
                                      usecols=[0, 1],
                                      names=['time', 'mains2'],
                                      dtype={'time': str},
                                      )
        app_df = pd.read_table(args.data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                   str(params_appliance[appliance_name]['channels']
                                       [params_appliance[appliance_name]['houses'].index(h)]) + '.dat',
                                   sep="\s+",
                                   nrows=nrows,
                                   usecols=[0, 1],
                                   names=['time', appliance_name],
                                   dtype={'time': str},
                                   )




        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')

        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)

        mains_df = mains1_df.join(mains2_df, how='outer')
        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
        mains_df.reset_index(inplace=True)


        # deleting original separate mains
        del mains_df['mains1'], mains_df['mains2']

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.show()

            # Appliance
            # app_df = app_df.set_index(app_df.columns[0])
            # app_df.index = pd.to_datetime(app_df.index, unit='s')
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
            # app_df.columns = [appliance_name]
        if debug:
            print("app_df:")
            print(app_df.head())
            plt.plot(app_df['time'], app_df[appliance_name])
            plt.show()

            # the timestamps of mains and appliance are not the same, we need to align them
            # 1. join the aggragte and appliance dataframes;
            # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        df_align = mains_df.join(app_df, how='outer'). \
                resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)


        del mains1_df, mains2_df, mains_df, app_df, df_align['time']

        mains = df_align['aggregate'].values
        app_data = df_align[appliance_name].values

        if debug:
                # plot the dtaset
            print("df_align:")
            print(df_align.head())
            plt.plot(df_align['aggregate'].values)
            plt.plot(df_align[appliance_name].values)
            plt.show()

        #output data
        df_align.to_csv(args.save_path + appliance_name +'_house_'+str(h)+'_training_.csv', mode='a', index=False, header=False)
        del df_align




if __name__ == '__main__':
    main()