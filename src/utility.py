"""
Utility module for common functions used in the project like loading, saving, plotting, etc.
"""
import glob
import os
import pickle
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


def load_config(yaml_file):
    '''
    load a yaml file and returns a dictionary
    '''
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return exc


def init_neptune(exp_name):
    """return neptune init object if you are using neptune
    """
    import neptune.new as neptune
    nt_config = ConfigParser()
    neptune_file = '.neptune_access.ini'
    print('Reading neptune config file: ', neptune_file)
    
    nt_config.read(neptune_file)
    project = nt_config['neptune_access']['project']
    api_token = nt_config['neptune_access']['api_token']

    nt_run = neptune.init(
        project=project,
        api_token=api_token, 
        name='ENViSEC',
        tags=exp_name)  # your neptune credentials

    # save configuration and module file to the neptune. 
    nt_run['configurations'].upload('config.yaml')
    nt_run['model_archs'].upload('src/models.py')
    nt_run['code'].upload('src/run.py')
    return nt_run

def utilize_gpu():
    """ 
    Utilise GPU if available, otherwise CPU
    """
    print('\n' + '+'*40)
    gpus = tf.config.list_physical_devices('GPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if gpus:
        # GPU support is recommended.
        use_gpu = True
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # # specify which GPU(s) to be used like "0,2,3,4"
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        # config1 = tf.compat.v1.ConfigProto()
        # config1.gpu_options.allow_growth = True
        # session = tf.compat.v1.Session(config=config1)
        print('GPU used: ', len(gpus))
    else:
        print('No GPU found!')
        use_gpu = False
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('+'*40)


# common settings for all the modules in the project
config = load_config('config.yaml')
debug = config['debug']
debug_rows = config['debug_rows']


def load_data(file):
    '''
    Load the data from the csv file
    '''
    print('-'*70)
    print(f'Loading data from: {file} ...')
    if debug:
        df = pd.read_csv(file, nrows=debug_rows)
    else:
        df = pd.read_csv(file, dtype='unicode', engine='c')
    print('Shape of the loaded data: ', df.shape)
    print('\nSample of the loaded data: \n', df.head(2))
    # print('info: ', df.info())
    print('-'*70)
    return df


def load_data_from_dir(data_dir):
    print('-'*60)
    print(f'Reading and combining the csv files of dir: {data_dir}...')
    csv_files = glob.glob(data_dir + '*.csv')
    df = pd.DataFrame()
    if debug:
        df = pd.read_csv(
            filepath_or_buffer=csv_files[0], 
            nrows=debug_rows)
    else:
        df = pd.concat(
            objs=[pd.read_csv(f) for f in csv_files], 
            ignore_index=True)
    print('Shape of the data: ', df.shape)
    print('\nSample of the loaded data: \n', df.head(2))
    print('-'*70)
    return df


def save_pickle(model, file):
    '''
    save the trained model
    '''
    with open(file, 'wb') as f:
        pickle.dump(model, f)
        # keras.models.save_pickle(model, file, save_format="h5")


def get_dataset_name(file):
    """
    returns dataset name from the input file
    """
    dt = file.rsplit('.', 1)[0].rsplit('/')
    name = dt[1] if len(dt)>0 else dt[0]
    return name


def plot_history(fig_name, history):
    plt.figure(figsize=(12, 10))
    plt.rcParams["font.size"] = "16"
    plt.xlabel('epoch')
    plt.ylabel('performance measures')
    # plt.plot(history.epoch, np.array(history.history['loss']),
    #          label='Train Loss')
    # plt.plot(history.epoch, np.array(history.history['val_loss']),
    #          label='Validation Loss')
    plt.plot(history.epoch, np.array(history.history['accuracy']),
             label='train_accuracy')
    plt.plot(history.epoch, np.array(history.history['val_accuracy']),
             label='val_accuracy')

    if 'precision' in history.history:
        plt.plot(history.epoch, np.array(history.history['precision']),
                 label='precision')
        plt.plot(history.epoch, np.array(history.history['val_precision']),
                 label='val_precision')

    if 'recall' in history.history:
        plt.plot(history.epoch, np.array(history.history['recall']),
                 label='recall')
        plt.plot(history.epoch, np.array(history.history['val_recall']),
                 label='val_recall')
    plt.legend()
    # plt.ylim([0, 1])
    plt.savefig(fig_name + '.pdf')

    # plotting loss curve separately
    plt.figure(figsize=(12, 10))
    plt.xlabel('epoch')
    plt.ylabel('performance measures')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='train_loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='val_loss')
    plt.legend()
    # plt.ylim([0, 1])
    plt.savefig(fig_name + '_loss.pdf')


def plot_curves(fig_name, kfold, trained_model, X, y, cv=3, return_times=True):
    # fig, ax1 = plt.subplots(1, 2, figsize=(10, 15))
    title = "Learning Curves"
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator=trained_model, X=X, y=y,
        cv=kfold, # cross validation default folds = 5 from sklearn 0.22
        return_times=return_times)
    plt.rcParams["font.size"] = "16"
    plt.plot(train_sizes, np.mean(train_scores, axis=1))
    plt.plot(train_sizes, np.mean(test_scores, axis=1))
    plt.legend(['Training', 'Testing'], loc='lower right')
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.savefig(fig_name)
