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
@File - to run a proof-of-concept IoT ML-model.
"""

from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import tensorflow as tf

from models import model_dnn
# local modules
from utility import plot_history

# Set seed for reproducibility, it fixes the randomness of the results.
# do not change this value for reproducibility, otherwise we will get different results.
seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)

# Training the TensorFlow ModelA
# Copied from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# This will convert our file of [1..16] strings to python lists to a numpy array that is TF friendly...


def file_to_np(filename):
    with open(filename, 'r') as f:
        f_lines = f.readlines()
    f_list = list(map(eval, f_lines))
    f_np = np.array([np.array(fi) for fi in f_list])
    return f_np

def init_neptune(exp_name):
    """return neptune init object if you are using neptune
    """
    import neptune.new as neptune
    nt_config = ConfigParser()
    neptune_access_file = '.neptune_access.ini'
    print('Reading neptune config file: ', neptune_access_file)
    nt_config.read(neptune_access_file)
    project = nt_config['neptune_access']['project']
    api_token = nt_config['neptune_access']['api_token']
    nt_run = neptune.init(
        project=project,
        api_token=api_token, 
        name='ENViSEC',
        tags=args.model_name)  # your neptune credentials
    return nt_run


def load_data(args):
    '''Loading the data'''
    num_samples = args.samples
    train_size = int(num_samples * 0.6)
    test_size = int(num_samples * 0.2)
    validate_size = int(num_samples * 0.2)

    # Load the data files..
    f_ledoff = file_to_np(args.data_dir + '/led-off2.log')[:num_samples]
    f_ledon = file_to_np(args.data_dir + '/led-on2.log')[:num_samples]

    # Setup the training data...
    train_data = []
    training_data = []
    training_labels = []
    train_data.append(f_ledoff[:train_size])
    train_data.append(f_ledon[:train_size])
    for i in range(2):
        for j in range(train_size):
            training_labels.append(i)
            training_data.append(train_data[i][j])

    training_data = np.array(training_data)
    training_labels = np.array(training_labels)

    # Set aside the test data - this will be used after training to see how we did...
    t_data = []
    test_data = []
    test_labels = []
    t_data.append(f_ledoff[train_size:(num_samples - validate_size)])
    t_data.append(f_ledon[train_size:(num_samples - validate_size)])
    for i in range(2):
        for j in range(test_size):
            test_labels.append(i)
            test_data.append(t_data[i][j])

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Shuffle the data sets so it isn't just 'all 1's then all 0's'...
    # This will help with the training, as our data is all linear for now
    training_data, training_labels = unison_shuffled_copies(
        training_data, training_labels)
    test_data, test_labels = unison_shuffled_copies(test_data, test_labels)
    return training_data, training_labels, test_data, test_labels


def train(args):
    # Load the data
    training_data, training_labels, test_data, test_labels = load_data(args)
    metrics = ['accuracy',
               'Precision',
               'Recall',
               ]
        
    # Setup the model
    model = model_dnn(args)
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=args.loss,
                  metrics=metrics,
                  )

   # Lets save our current model state so we can reload it later
    model.save_weights(args.model_dir + 'pre-fit.weights')
    
    tf_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=35, 
                                         monitor='val_loss',
                                         mode='min',
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='results/basic_dnn/model.{epoch:02d}-{val_loss:.2f}.h5',
                                           save_best_only=True,
                                           monitor='val_loss',
                                           mode='min',),
        tf.keras.callbacks.TensorBoard(log_dir='results/logs')]
    
    if args.use_neptune:
        from neptune.new.integrations.tensorflow_keras import NeptuneCallback
        print('\n' + '-' * 30 + 'Neptune' + '-' * 30 + '\n')
        nt_run = init_neptune('ENViSEC')
        neptune_cbk = NeptuneCallback(run=nt_run, base_namespace='metrics')
        # nt_run.send_artifact(args.model_dir + 'pre-fit.weights')
        # nt_run.send_artifact(args.model_dir + 'basic_dnn.h5')
        # nt_run.send_artifact(args.model_dir + 'history.png')
        # nt_run.send_artifact(args.model_dir + 'history.json')
        nt_run['parameters'] = vars(args)
        # nt_run['performance_metrics'] = history.history
        tf_callbacks.append(neptune_cbk)
        # nt_run.stop()
    
    # fit the model
    history = model.fit(training_data,
                        training_labels,
                        epochs=int(args.epochs),
                        validation_split=0.2,
                        verbose=2, 
                        callbacks=[tf_callbacks])

    # Plot the training history
    plot_history(args, history)

    # Save the model
    model.save(args.model_dir + 'basic_dnn.h5')
    return model, history


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train",
                        default=False, help="Train mode")
    parser.add_argument("-p", "--predict", dest="predict",
                        default=False, help="Test/Predict mode")

    parser.add_argument("-m", "--model_name", dest="model_name",
                        default="basic_dnn", help="Model to use")
    parser.add_argument("-e", "--epochs", dest="epochs",
                        default=1500, help="Number of epochs to train")
    parser.add_argument("-s", "--samples", dest="samples",
                        default=5000, help="Number of samples to train")
    parser.add_argument("-v", "--validate", dest="validate",
                        default=2000, help="Number of samples to validate")
    parser.add_argument("-b", "--batch", dest="batch",
                        default=32, help="Batch size")
    parser.add_argument("-l", "--loss", dest="loss",
                        default="sparse_categorical_crossentropy", help="Loss function")
    parser.add_argument("-lr", "--lr", dest="lr",
                        default=0.001, help="Learning rate")
    parser.add_argument("-o", "--optimizer", dest="optimizer",
                        default="Adam", help="Optimizer")

    parser.add_argument("-d", "--debug", dest="debug",
                        default=False, help="Debug mode")
    parser.add_argument("-dd", "--data_dir", dest="data_dir",
                        default='data/processed/', help="Input mode")
    parser.add_argument("-f", "--filter", dest="filter",
                        default=False, help="Filter mode")
    parser.add_argument("-nep", "--use_neptune", dest="use_neptune",
                        default=True, help="Filter mode")
    args = parser.parse_args()

    args.model_dir = 'models/' + args.model_name + '/'

    if args.debug:  # Debug mode for sample testing/testing
        args.epochs = 10
        args.samples = 100

    print("Type of data_dir", type(args.data_dir))

    # -------------------------------------------Train/Validate---------------------------------------------------------------
    if args.train:
        print('\n' + '-' * 30 + 'Predict/Test' + '-' * 30 + '\n')
        model, history = train(args)

    # -------------------------------------------Predict/Test---------------------------------------------------------------
    if args.predict:  # Test mode for sample testing
        _, _, test_data, test_labels = load_data(args)
        print('\n\n' + '-' * 30 + 'Predict/Test' + '-' * 30 + '\n')
        eval_metrics = model.evaluate(
            test_data, test_labels, return_dict=True, verbose=2)
        print('\nEvaluation metrics on testdata:')
        for metric_name in eval_metrics:
            print('\t{}: {} %'.format(metric_name,
                  eval_metrics[metric_name] * 100))
        print('\n' + '-' * 70 + '\n')
