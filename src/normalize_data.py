"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT

Project: ENViSEC - Artificial Intelligence-enabled Cybersecurity for Future Smart Environments 
(funded from the European Union’s Horizon 2020, NGI-POINTER under grant agreement No 871528).
@Authors: Guru Bhandari, Andreas Lyth, Andrii Shalaginov, Tor-Morten Grønli
@Programmer: Guru Bhandari
@File - normalization of the data.
"""
import gc
import os
from pathlib import Path

# import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

from src.utility import load_config, load_data, load_data_from_dir

gc.collect()
# pd.set_option('display.precision', 3)


# def convert_ip2int(ip_add):
#     '''
#     converts ip address to equivalent integer value
#     '''
#     # ip_ver = 4 if '.' in ip_add else 6
#     try:
#         if '.' in ip_add:  # ipv4
#             ip = ipaddress.IPv4Address(ip_add)
#         elif ':' in ip_add:  # ipv6
#             ip = ipaddress.IPv6Address(ip_add)
#         else:
#             ip = 0  # 0 refers invalid ip address
#     except Exception as e:
#         # print('Invalid IP address: ', e)
#         ip = 0  # 0 refers invalid ip address
#     return int(ip)


def normalize_data(df):
    '''
    Normalize data using MinMaxScaler for X and return y as it is.
    '''
    print('Sample data:\n', df.head(2))
    x = df.values  # returns a numpy array
    print('-'*60)
    print('\nShape of the data before normalization: ', df.shape)
    print('x.dtype: ', x.dtype)
    print('\nSample X: \n', x[:2])
    print('-'*60)
    print('\n' + '-'*20 + 'Normalizing' + '-'*20)
    print('\nNormalizing data using MinMaxScaler...\n')
    print('-'*60)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    print('Completed fitting!')
    df_scaled = pd.DataFrame(x_scaled)
    df_scaled.columns = df.columns
    return df_scaled


def filter_columns(df, txt_file=None):
    '''
    remove the unique columns from the dataframe because 
    they will not contribute for the machine learning model.
    return: the list of all unique values and single value columns in the dataframe
    '''
    cols_to_drop = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique == 1 or n_unique == len(df):
            cols_to_drop.append(col)
    print('Columns to remove from the dataset based on number of unique items \
        (1 or all unique):', cols_to_drop)
    df = df.drop(cols_to_drop, axis=1)
    return df, cols_to_drop


def replace_values(df, cat_cols, data_process_for='train'):
    '''
    replaces the values in the dataframe with some relevant values.
    '''
    dict_rep = {
        '_':0, 
        '-':0,
        '(empty)':1  # make it different than '-' for 'tunnel_parents'
        }
    print('-' * 100)
    print('Replacing the unknown or NaNs values with 0...')

    if 'service' in df.columns:
        df['service'] = df['service'].astype(str).str.replace('-', 'unknown') 
    
    non_cat_cols = ['multi_label', 'ts', 'label']
    # print('describe before replacing: ', df.describe())
    print('-' * 100)
    # print('Info before replacing: ', df.info())
    # cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    # raises an error- terminated by signal SIGKILL (forced quit)
    # cat_cols = [col for col in cat_cols if col not in non_cat_cols] 

    # encode categorical variabless
    if cat_cols:
        df = pd.get_dummies(data=df, columns=cat_cols, sparse=True) 

    print('Categorical columns to be encoded are: ', cat_cols)
    # Replacing '-' with other values
    # <guru> Need to check if this is the best way to do this with IP address columns?

    # remove rows which has unknown value (-) in the following columns. 
    if 'duration' in df.columns:
        # df = df[df.duration!='-']
        df['duration'] = df['duration'].replace(dict_rep).astype(float)

    if 'orig_bytes' in df.columns:
        df['orig_bytes'] = df['orig_bytes'].replace(dict_rep, regex=True).astype(float)
        # df = df[df.orig_bytes!='-']

    if 'resp_bytes' in df.columns:
        df['resp_bytes'] = df['resp_bytes'].replace(dict_rep, regex=True).astype(float)
        # df = df[df.resp_bytes!='-']

    if 'tunnel_parents' in df.columns:
        # <guru> more than 99% samples have: '-': 1794822 and (empty):83921
        if data_process_for=='train':
            df = df[df.tunnel_parents.isin(['-','(empty)'])] # filter other than '-' and '(empty)'.
        df['tunnel_parents'] = df['tunnel_parents'].astype(str).str.strip().replace(dict_rep).astype(float)
        
    # correcting labels
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower().replace(
            {'malicious':1, 'benign':0}).astype(int)
    if 'multi_label' in df.columns:
        # '-' is represented for benign packets
        df['multi_label'] = np.where(
            (df['multi_label'] == '-'), 'benign', df['multi_label']) 

    # # correcting IP address fields, replacing _ by . to convert it into interger. 
    # if 'id_orig_h' in df.columns:
    #     df['id_orig_h'] = df['id_orig_h'].astype(str).str.replace('_', '.')
    #     df['id_resp_h'] = df.id_resp_h.astype(str).apply(convert_ip2int) 

    # if 'id_resp_h' in df.columns:
    #     df['id_resp_h'] = df['id_resp_h'].astype(str).str.replace('_', '.') 
    #     df['id_orig_h'] = df.id_orig_h.astype(str).apply(convert_ip2int)
    return df


def render_plots(df):
    '''
    shows the distribution of the dataframe
    '''
    FIG_DIR = 'results/data-plots/'
    df.multi_label.value_counts().plot.pie(subplots=True,
                                           shadow=True,
                                           autopct='%1.1f%%',
                                           figsize=(6, 6),
                                           labeldistance=1.2,
                                           legend=True,
                                           title='Distribution of Malicious/Benign Traffic'
                                           )
    plt.savefig(FIG_DIR + 'distribution_pie.pdf', format='pdf')
    # Using Pearson Correlation
    plt.figure(figsize=(20, 18))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds,
                fmt='.2f', annot_kws={'size': 8})
    plt.title('Heatmap showing correlation on IoT-23 dataset features', fontsize=14)
    plt.savefig(FIG_DIR + 'iot-23-heatmap.pdf', format='pdf')
    # plt.show()


def perform_norm(raw_data, normalize_file):
    """
    perform normalization of the given raw_data file/dir of csv files. 
    """
    if os.path.isdir(raw_data):
        df = load_data_from_dir(raw_data)
        # save the combined dataframe into a CSV fle. 
        merged_csv = config['data_dir'] + 'normalized/IoT-23.csv'
        df.to_csv(merged_csv, index=False) 
        print('Saved the combined raw data at: ', merged_csv)
        print('-'*60)
    elif os.path.isfile(raw_data):
        df = load_data(raw_data)
    else:
        print('Unknown raw data!')
        exit(1)

    df = df.rename(columns={'detailed-label': 'multi_label'})

    if 'multi_label' in df.columns:
        df['multi_label'] = df['multi_label'].replace({'-':'benign'}) # correct label
    df, cols_removed = filter_columns(df)

    cat_cols = ['proto', 'service', 'conn_state']

    if config['preprocess']['include_history']:
        cat_cols.append('history')
    else:
        df = df.drop('history', 1) # remove history if disabled

    cat_cols = [col for col in cat_cols if col not in cols_removed]

    # replacing the values
    df = replace_values(df, cat_cols)
    # df.to_csv('composed_file.csv', header=False)
    
    # normalizing the data using MinMaxScaler except for the label columns
    normalize_cols = [col for col in df.columns if col not in ['label', 'multi_label']]
    df[normalize_cols] = normalize_data(df[normalize_cols])

    for col in normalize_cols:
        df[col]=df[col].astype(float)
    # df[normalize_cols]=(df[normalize_cols]-df[normalize_cols].min())/(df[normalize_cols].max()-df[normalize_cols].min())
    
    print('Shape of the normalized data: ', df.shape)
    print('\nSample data after normalization:\n')
    print(df.head(2))
    print('-'*100)
    # print('Ploting the hist of the normalized data...')
    # df.plot.hist()
    # # rendering the plots
    # print('Ploting the heatmap of the normalized data...')
    # render_plots(df)

    # saving the dataframe into csv file
    df.to_csv(normalize_file, index=False)
    print('Normalized dataset is saved at :', normalize_file)
    print('-'*100)


if __name__ == '__main__':
    config = load_config('config.yaml')
    raw_data = config['data_dir'] + config['preprocess']['raw_data']
    normalize_file = config['data_dir'] + config['preprocess']['normalized_data']
    FIG_DIR = config['result_dir'] + 'data_plots/'

    # create folders if not exist
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(normalize_file.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    # perform normalization
    perform_norm(raw_data, normalize_file)
