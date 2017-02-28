#!/usr/local/bin/python3

# Modules

import numpy as np
import pandas as pd
import os
import math as mt
import random
import pickle
import gzip

from network import Network
# Functions

# MNIST data
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    
# Own functions

def import_data(file_loc, separator, id_colname):
    dataset = pd.read_csv(filepath_or_buffer=file_loc, sep=separator)

    idcol = dataset[id_colname]
    dataset = dataset.drop(id_colname, 1)

    print("The dataset contains %s columns and %s rows." % (dataset.shape[1], dataset.shape[0]))
    print("See preview of data below.")
    print(dataset.head(5))

    return dataset, idcol

def make_data_frame(input_data, id_col_df):
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        df = pd.DataFrame(input_data)

    df.set_index(id_col_df,
                 drop=True,
                 append=False,
                 inplace=True,
                 verify_integrity=False
                 )

    return df


def categorical_to_numeric(categorical_data):
    n_classes = len(pd.unique(categorical_data))

    temp_data = pd.get_dummies(categorical_data)
    categories = np.arange(n_classes)

    numeric_data = np.dot(temp_data, categories)
    numeric_data = numeric_data.astype(int)

    return numeric_data


def train_test_splitter(target, features, id_column, train_size):
    y_full = make_data_frame(input_data=target,
                             id_col_df=id_column
                             )
    y_full.columns = ['y']

    x_full = make_data_frame(input_data=features,
                             id_col_df=id_column
                             )

    full = y_full.join(x_full)
    classes = pd.unique(full.iloc[:, 0])
    train_data = pd.DataFrame([])

    for c in classes:
        class_full = len(full[full['y'] == c])
        class_train_data = full[full['y'] == c].iloc[:int(mt.floor(train_size * class_full)), :]
        train_data = train_data.append(class_train_data)

    test_data = full[~full.index.isin(train_data.index)]

    y_train = train_data.iloc[:, 0]
    x_train = train_data.iloc[:, 1:train_data.shape[1]]

    y_test = test_data.iloc[:, 0]
    x_test = test_data.iloc[:, 1:test_data.shape[1]]

    return y_train, x_train, y_test, x_test


def one_hot_encoder(vector):
    vector_to_matrix = pd.get_dummies(vector)
    return vector_to_matrix


# Neural Network





    # Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


    
if __name__ == "__main__":
    # Script

    work_dir = "/Users/cornelisvletter/Desktop/Programming/NeuralNetworks"
    filename = "/iris.csv"
    file_loc = work_dir + filename
    os.chdir(work_dir)


    """
    #Alternative Dataset
    raw_data, id_col = import_data(file_loc=file_loc,
                                   separator=",",
                                   id_colname="Id"
                                   )

    raw_features = raw_data.iloc[:, 0:4]
    raw_target = categorical_to_numeric(categorical_data=raw_data["Species"])

    y_train_1d, x_train, y_test_1d, x_test = train_test_splitter(target=raw_target,
                                                           features=raw_features,
                                                           id_column=id_col,
                                                           train_size=0.7
                                                           )

    y_train = one_hot_encoder(vector=y_train_1d)
    y_test = one_hot_encoder(vector=y_test_1d)
    """

    #MNIST Data
    training_data, validation_data, test_data = load_data_wrapper()

    net = Network([784, 30, 10])
    net.SGD(training_data, 3, 10, 3.0, test_data=test_data)

    weights = net.weights
    biases = net.biases
