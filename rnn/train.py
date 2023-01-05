"""
Import Libraries
"""
import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from training_dependencies import make_model
from training_dependencies import TimeCallback
from training_dependencies import CheckpointModel
from load_data import load_experiment


def train(training_args, pid):
    architecture = training_args.architecture
    learning_rate = training_args.learning_rate
    hid_dim = training_args.hid_dim
    epochs = training_args.epochs
    dataset = training_args.dataset
    permute = training_args.permute
    pad = training_args.pad
    orientation = training_args.orientation
    ensemble_end_time = training_args.ensemble_end_time

    # fix to adjoint training
    penalty_weight = 0.0

    # generate experiment data
    (x_train, y_train), (x_test, y_test) = load_experiment(dataset, permute, pad, orientation)

    if architecture == 'unitary':
        x_train = tf.cast(x_train, dtype=tf.dtypes.complex64)
        x_test = tf.cast(x_test, dtype=tf.dtypes.complex64)

    T = int(x_train.shape[1])
    ft_dim = 50 if dataset in ['imdb.npz', 'reuters.npz'] else int(x_train.shape[2])
    output_dim = 1 if dataset in ['imdb.npz', 'adding'] else int(len(np.unique(y_train)))
    embed = True if dataset in ['imdb.npz', 'reuters.npz'] else False

    # build model
    model = make_model(architecture, T, ft_dim, hid_dim, output_dim, penalty_weight, learning_rate, embed=embed)

    # callbacks
    time_callback = TimeCallback()
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()
    model_chk = CheckpointModel(training_args, pid)

    # fit model
    model_out = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test,y_test), callbacks=[time_callback, nan_terminate, model_chk], verbose=0)

    # store model/training information in history
    model_out.history['model_type'] = architecture
    model_out.history['dataset'] = dataset
    model_out.history['permute'] = permute
    model_out.history['pad'] = pad
    model_out.history['orientation'] = orientation
    model_out.history['hid_dim'] = hid_dim
    model_out.history['penalty_weight'] = penalty_weight
    model_out.history['epochs'] = epochs
    model_out.history['learning_rate'] = learning_rate
    model_out.history['epoch_time'] = time_callback.times
    model_out.history['num_parameters'] = int(np.sum([K.count_params(w) for w in model.trainable_weights]))

    return model_out.history

def str_to_bool(bool_string):
    out = True if bool_string == 'True' else False
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('hid_dim', type=int)
    parser.add_argument('epochs', type=int)
    parser.add_argument('dataset', type=str)
    parser.add_argument('permute', type=str_to_bool)
    parser.add_argument('pad', type=int)
    parser.add_argument('orientation', type=str)
    parser.add_argument('identifier', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('ensemble_end_time', type=float)

    return parser.parse_args()


def main():

    num_omp = intra = int(os.environ["OMP_NUM_THREADS"])
    pid = os.getpid()
    inter = 2

    tf.config.threading.set_intra_op_parallelism_threads(num_omp)
    tf.config.threading.set_inter_op_parallelism_threads(inter)

    args = parse_args()
    training_history = train(args, pid)

    output_file_path = os.path.join(args.path, f'training-history-{args.identifier}-{pid}')
    np.save(output_file_path, training_history)
    time.sleep(1)
    hist_exist = os.path.exists(f'{output_file_path}.npy')

    # remove any checkpoint files on completion
    chk_path = f'/grand/rnn-robustness/test-checkpoints/model-{args.identifier}-{pid}'
    chk_exist = os.path.exists(chk_path)
    if chk_exist and hist_exist:
        os.remove(chk_path)

    return



if __name__== "__main__":
    main()
