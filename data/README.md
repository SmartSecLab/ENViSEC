# Datasets used for the experiment

### Training and testing data:
We have taken _[Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23)_ and _[Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications#files)_
datasets for training and testing the machine learning models.
The _IoT-23_ dataset is semi-structured logging information of the packets labeled with malicious and benign IoT network traffic. The dataset was originally created by Avast AIC laboratory collecting from different IoT devices. Another dataset _Edge-IIoTset_ is already pre-processed. However, we have carried out further pre-processing and cleaning steps.

_The original dataset path for the project:_

```
DATA_DIR = 'data/iot_23_datasets_small/'
```

_The generated csv files (raw without filtering) generated from original textual files:_

```
DATA_DIR = 'data/processed/<processed_data_file>.csv'
```

_The processed data path after preprocessing/filtering and feature selection:_

```
DATA_DIR = 'data/processed/<processed_data_file>.csv'
```

### Prediction data:
The predicting data is the locally running network traffic data from the  IoT test cases of nodes in local smart environments. Configure __predict.yaml__ file to retrieve the prediction data and apply the trained model.
The network traffic data using __zeek__ can be given as providing __log_file__ parameter value:
```
log_file: data/predict/conn.log
```
