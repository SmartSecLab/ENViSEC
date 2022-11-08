import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import schedule
import yaml
from flask import Flask, jsonify
from IPython.display import display
from sklearn import preprocessing
from src.utility import load_config
from tensorflow.keras.models import load_model

app = Flask(__name__)   # create an instance of the Flask class
# data_path = '/data/'
# pred_config = 'predict.yaml'
# config = load_config(pred_config)

from src.normalize_data import replace_values
from src.utility import load_config


config = load_config('predict.yaml')    
scan_time = int(config['time'])
pred_config = 'predict.yaml'
data_path = '/data/'
    

def update_hosturl(pred_config, hosturl):
    """ Update the hosturl in the config file, 
    so that the server can be accessed from the nodes 
    to get rid of manually changing the hosturl and port every time. 
    Args:
        pred_config_file: str: path to the predict config file
        hosturl: str: hosturl to be updated in the config file
    """  
    config = load_config(pred_config)
    config['host'] = hosturl
    print('\nLocalhost URL to run in node-side:')
    print('='*30)
    print(config['host'])
    print('='*30)
    with open(pred_config, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        print(f'Updated prediction config file: {pred_config}\n')
        

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
        self.df, self.df_X = self.load_log2df()
        self.df = self.refine_data(self.df)
        return (self.df, self.df_X)

    #     print('Loading trained model is located at: ', self.model_path, '...\n')
    #     model = load_model(filepath=self.model_path)
    #     X =  np.asarray(df.values).astype(np.float32)

    #     # preds = model.predict(X, verbose=0)
    #     preds = model.predict(X)

    #     encoder = preprocessing.LabelEncoder() # need to check
    #     encoder.classes_ = np.load(self.classes_file, allow_pickle=True)
    #     preds_index = [np.argmax(y, axis=None, out=None) for y in preds]
    #     df_X['multi_label'] = list(encoder.inverse_transform(preds_index))
        
    #     count_benign = (df_X.multi_label=='benign').sum()
    #     count_attack = len(df_X) - count_benign
        
    #     print('-'*50)
    #     print('The Prediction Result:')
    #     print('-'*50)
    #     print('\t #Predicted Benign:  ', count_benign)
    #     print('\t #Predicted Attacks: ', count_attack)
    #     print(f'\t Benign Ratio: {(count_benign/(count_benign + count_attack)).round(2)* 100} %')
    #     print('-'*50)
    #     print('-'*50)
    #     df_X_summary = df_X[['id_orig_h', 'id_orig_p', 'id_resp_h', 'id_resp_p',
    #         'proto', 'service', 'multi_label']]
    #     df_attack = df_X_summary[df_X_summary.multi_label!='benign'].reset_index(drop=True)

    #     if len(df_attack)>0:
    #         print('Please check the following suspected devices (predicted as malicious):') 
    #         print('-'*50)
    #         display(df_attack)
    #         self.predict_output = self.predicted_path + 'report_' + self.time_stamp + '.csv'
    #         df_attack.to_csv(self.predict_output, index=False)


@app.route(data_path, methods=['GET', 'POST']) 
def job():
    print('\n\n' + '='*60)
    # : is not allowed in file name by OneDrive
    time_stamp = str(datetime.now().strftime(f'%Y-%m-%d_%H:%M:%S')).replace(':', '.') 
    print('Scanning the network at time: ', time_stamp)
    print('='*60)
    pred = Predict(config)
    df, df_X = pred.predict_label() # predict the log file
    # print('\n df:\n', df)
    # print('\n df_X:\n', df_X)
    return df.to_json()
        

if __name__ == '__main__':
    pred = Predict(config)
    schedule.every(scan_time).seconds.do(job)
    
    while 1:
        schedule.run_pending()
        time.sleep(1)
        rand_num = randint(5000, 7000)
        host_url = 'http://localhost:' + str(rand_num ) + data_path
        # host_url = 'http://127.0.0.1:' + str(rand_num ) + data_path
        print("Host URL: " + host_url)
        update_hosturl(pred_config, host_url)
        app.run(host='0.0.0.0', port=rand_num )
