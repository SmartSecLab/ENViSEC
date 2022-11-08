import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread, Timer
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import schedule
from IPython.display import display
from sklearn import preprocessing
from src.utility import load_config
from tensorflow.keras.models import load_model
from io import BytesIO


def get_response(host_url):
  """
  get the response from the url
  """
  # host_url = config['host']
  # time_sec = config['time'] # run itself again after specified seconds
  # timer = Timer(time_sec, get_response, args=(host_url,)).start()
  print('host_url: ', host_url)
  feed = urlopen(host_url)
  # feed = urlretrieve(host_url)
  
  df = pd.read_json(BytesIO(feed.read()))
  print('-'*20 + 'Processed data from the host: ' + '-'*20)
  print(df.head(5))
  print('-'*20)
  return df
        
class Fetcher():
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

    def predict_attack(self, df):
          '''
          predict the multi-level attacks of the dataframe.
          '''
          identifier_cols = ['id_orig_h', 'id_orig_p', 'id_resp_h', 'id_resp_p',
              'proto', 'service', 'multi_label']
          
          print('Loading trained model is located at: ', self.model_path, '...\n')
          model = load_model(filepath=self.model_path)
          X =  np.asarray(df.values).astype(np.float32)

          # preds = model.predict(X, verbose=0)
          preds = model.predict(X)

          encoder = preprocessing.LabelEncoder() # need to check
          encoder.classes_ = np.load(self.classes_file, allow_pickle=True)
          pred_index = [np.argmax(y, axis=None, out=None) for y in preds]
          print('\n df columns: ', df.columns)
          
          # get predicted label transforming the pred_index
          df['multi_label'] = list(encoder.inverse_transform(pred_index))
          
          count_benign = (df.multi_label=='benign').sum()
          count_attack = len(df) - count_benign
          
          print('-'*50 + '\nPrediction Result:\n' + '-'*50)
          print('\t #Predicted Benign:  ', count_benign)
          print('\t #Predicted Attacks: ', count_attack)
          print(f'\t Benign Ratio: {(count_benign/(count_benign + count_attack)).round(2)* 100} %')
          print('-'*50 + '\n' + '-'*50)
          
          # add these columns
          # df = df[identifier_cols]
          df_attack = df[df.multi_label!='benign'].reset_index(drop=True)
          self.time_stamp = str(datetime.now().strftime(f'%Y-%m-%d_%H:%M:%S')).replace(':', '.') 

          if len(df_attack)>0:
              print('Please check the following suspected devices (predicted as malicious):') 
              print('-'*50)
              display(df_attack)
              self.predict_output = self.predicted_path + 'report_' + self.time_stamp + '.csv'
              df_attack.to_csv(self.predict_output, index=False)


# config = load_config('predict.yaml')
# df = get_response(config['host'])
# Fetcher(config).predict_attack(df)

# predict_attack(df)

if __name__ == '__main__':
    config = load_config('predict.yaml')    
    
    # scan_time = int(config['time'])
    # pred = Predict(config)
    # schedule.every(scan_time).seconds.do(pred.job)

    while 1:
        # schedule.run_pending()
        # time.sleep(1)
        df = get_response(config['host'])
        Fetcher(config).predict_attack(df)
        time.sleep(config['time'])