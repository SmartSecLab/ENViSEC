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
@File - prediction on the specified data.
"""

import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import schedule
from IPython.display import display
from sklearn import preprocessing
from tensorflow.keras.models import load_model

from src.normalize_data import replace_values
from src.utility import load_config

class Predict:
    def __init__(self, config):
        self.log_file = config['log_file']
        self.train_data = config['train_data']
        self.model_path = config['model_path']
        self.classes_file = config['classes_file']
        self.predicted_path = config['output']
        
        if config['clean']:
            print('Cleaning the already predicted output...')
            if Path(self.predicted_path).is_dir():
                shutil.rmtree(self.predicted_path)
        Path(self.predicted_path).mkdir(parents=True, exist_ok=True)

    def converter(self, val):
        """ correcting format of the data """
        if len(val) > 0 and (val!= '#' or val!=' '):
            return val.replace('#', '').replace('.', '_').strip()
        else:
            return None

    def load_log2df(self):
        """ 
        load the log file into a dataframe
        """
        df = pd.read_table(self.log_file, sep='\t', skiprows=6)
        df = df.drop(df.index[0], axis='index').reset_index(drop=True)
        df.rename(columns=dict(zip(list(df.columns[0:-1]), list(df.columns[1:]))), inplace=True)
        df = df.rename(columns=self.converter)

        # make a copy to identify the devices after prediction by its IP address
        df_X = df.copy()
        df = df.iloc[: , :-1]
        cat_cols = ['proto', 'service', 'conn_state', 'history']
        df = df.replace('-', 0).fillna(0)
        df = replace_values(df, cat_cols, data_process_for='predict')
        return df, df_X
        
        
    def refine_data(self, df):
        """
        select the columns based on the processed trained data. 
        """
        df_train = pd.read_csv(self.train_data, nrows=2)
        selected_cols = list(set(df.columns).intersection(set(df_train.columns)))
        df = df[selected_cols]

        fill_cols = list(set(list(df_train.columns)).difference(set(list(df.columns))))
        label_cols = ['label','multi_label']
        fill_cols = [x for x in fill_cols if x not in label_cols]
        # print('Columns in predicted data: ', df.columns)
        df[fill_cols] = 0  # fill remaining columns with default value 0

        assert len(df_train.columns) == (len(df.columns) + len(label_cols)), \
            'Predicted data should have same columns as of trained data!'

        print('Input data shape: ', df.shape)
        print('Columns in predicted data: ', df.columns)
        return df
    
    
    def predict_label(self):
        """ 
        predict the data and save the results in the output folder.
        """
        df, df_X = self.load_log2df()
        df = self.refine_data(df)

        print('Loading trained model is located at: ', self.model_path, '...\n')
        model = load_model(filepath=self.model_path)
        X =  np.asarray(df.values).astype(np.float32)

        # preds = model.predict(X, verbose=0)
        preds = model.predict(X)

        encoder = preprocessing.LabelEncoder() # need to check
        encoder.classes_ = np.load(self.classes_file, allow_pickle=True)
        preds_index = [np.argmax(y, axis=None, out=None) for y in preds]
        df_X['multi_label'] = list(encoder.inverse_transform(preds_index))
        
        count_benign = (df_X.multi_label=='benign').sum()
        count_attack = len(df_X) - count_benign
        
        print('-'*50)
        print('Prediction Result:')
        print('-'*50)
        print('\t #Predicted Benign:  ', count_benign)
        print('\t #Predicted Attacks: ', count_attack)
        print(f'\t Benign Ratio: {(count_benign/(count_benign + count_attack)).round(2)* 100} %')
        print('-'*50)
        print('-'*50)
        df_X_summary = df_X[['id_orig_h', 'id_orig_p', 'id_resp_h', 'id_resp_p',
            'proto', 'service', 'multi_label']]
        df_attack = df_X_summary[df_X_summary.multi_label!='benign'].reset_index(drop=True)

        if len(df_attack)>0:
            print('Please check the following suspected devices (predicted as malicious):') 
            print('-'*50)
            display(df_attack)
            self.predict_output = self.predicted_path + 'report_' + self.time_stamp + '.csv'
            df_attack.to_csv(self.predict_output, index=False)

    def job(self):
        print('\n\n' + '='*60)
        # : is not allowed in file name by OneDrive
        self.time_stamp = str(datetime.now().strftime(f'%Y-%m-%d_%H:%M:%S')).replace(':', '.') 
        print('Scanning the network at time: ', self.time_stamp)
        print('='*60)
        self.predict_label() # predict the log file


if __name__ == '__main__':
    config = load_config('predict.yaml')    
    scan_time = int(config['time'])
    pred = Predict(config)
    schedule.every(scan_time).seconds.do(pred.job)

    while 1:
        schedule.run_pending()
        time.sleep(1)
