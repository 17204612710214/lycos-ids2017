#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Filename: deeplearning.py
Date: Nov 10, 2019 - 10:07
Name: Anonymous
Description:
    -  Deep learning library
"""

import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from copy import deepcopy
from sklearn.metrics import confusion_matrix
# reduce debug info from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# MLP training is faster without CUDA / GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# set seed so that we can reproduce exact same settings
tf.random.set_seed(0)

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './tf_logs/gradient_tape/' + current_time + '/train'
crossval_log_dir = './tf_logs/gradient_tape/' + current_time + '/crossval'


class Dataset(object):
    """
    Dataset class
    """

    def __init__(self):
        """
        Dataset creation

        Parameter
        ---------
        None

        Return
        ------
        Created object
        """
        self.data_file = ''
        self.data = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.labels_onehot = pd.DataFrame()
        self.df = pd.DataFrame()
        self.df_perm = pd.DataFrame()  # for debug purpose
        self.n_samples = 0
        self.n_features = 0
        self.label_dict = {}
        self.inv_label_dict = {}
        self.n_classes = 0
        self.batch_size = 0
        self.n_batch = 0
        self.current_batch_idx = 0
        self.true_distribution = []

    def from_file(self, data_file, file_format, size_percent):
        """
        Load data file and if needed label file

        Parameters
        ----------
        data_file: string
            filename (incl. path) for data
        file_format: string
            supported format: 'csv' and 'parquet'
        size_percent: int
            percentage of the file to be loaded (from 1 to 100

        Returns
        -------
        None
        """
        self.data_file = data_file
        df = pd.DataFrame()
        if file_format == 'csv':
            df = self._csv2df(data_file)
        elif file_format == 'parquet':
            df = self._pq2df(data_file)
        else:
            print("[ERROR] Format {} not supported".format(format))
        max_idx = int(df.shape[0] * size_percent / 100)
        self.df = df[:max_idx][:]
        self.df_perm = self.df
        self.data = self.df
        self.df.name = data_file
        self.n_samples = np.shape(self.df)[0]
        self.n_features = np.shape(self.df)[1]

    @staticmethod
    def _csv2df(data_file):
        """
        Load CSV file in pandas DataFrame

        Parameter
        ---------
        self: Dataset

        Return
        ------
        df: DataFrame
            Loaded data Frame
        """
        df = pd.read_csv(data_file, encoding="ISO-8859-1", low_memory=False)
        return df

    @staticmethod
    def _pq2df(data_file):
        """
        Load parquet file in pandas DataFrame

        Parameter
        ---------
        self: Dataset

        Return
        ------
        df: DataFrame
            Loaded data Frame
        """
        df = pd.read_parquet(data_file)
        return df

    def drop_dfcol(self, drop_list):
        """
        Remove the columns from dataset DataFrame that are specified in
        drop_list and convert to numpy array loaded in data

        Parameters:
        -----------
        drop_list: list
            list of column names

        Return:
        -------
        None
        """
        self.data = self.df
        for lbl in drop_list:
            self.data = self.data.drop(lbl, axis=1)
        self.n_features = np.shape(self.data)[1]

    def select_dfcol_as_label(self, col_name, bin_class):
        """
        Select the column from dataset DataFrame and specified in col_name as
        label

        Parameters
        ----------
        col_name: string
            name of the column to select
        bin_class: bool
            convert multiclass label to enable binary classification

        Returns
        -------
        None

        """

        self.labels = self.df[col_name]
        if bin_class is True:
            print("not supported yet")
        else:
            self.label_dict = {label: idx for idx, label in enumerate(
                np.unique(self.df[col_name]))}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        self.labels.replace(to_replace=self.label_dict, value=None,
                            inplace=True)
        if bin_class is True:
            print("not supported yet")
        self.n_classes = np.size(self.labels.value_counts().index)
        self.labels_onehot = self.onehot_encode(self.labels, self.n_classes)
        # key_list = []
        # for key in dict.keys(self.label_dict):
        #     key_list.append(key)
        # self.df.reset_index(key_list)

        self.true_distribution = self._get_true_distribution()

    def onehot_encode(self, df, n_classes):
        """
        one-hot encoding

        Parameters:
        -----------
        df: DataFrame
            pandas DataFrame containing list of labels (numeric values) to be
            encoded

        Return:
        -------
        onehot_df: DataFrame
            m by n_classes matrix (m being the size of labels)
        """
        np_array = df.values
        m = np.size(np_array)
        n = n_classes
        onehot_matrix = np.zeros((m, n))
        for i in range(0, m):
            onehot_matrix[i][np_array[i]] = 1
        onehot_df = pd.DataFrame(np.int_(onehot_matrix),
                                 columns=self.label_dict)
        return onehot_df

    def set_batch_size(self, batch_size):
        """
        Function setting the batch size

        Parameters:
        -----------
        batch_size: integer
            value corresponding to the number of training examples contained
            in a batch

        Return:
        -------
        None

        """
        self.batch_size = batch_size
        self.n_batch = math.ceil(self.n_samples / batch_size)

    def get_next_batch(self, onehot=True):
        """
        Function returning the next batch
        When dataset has been fully fetched, a permutation is done
        By default, the labels will be one-hot encoded

        Parameters:
        -----------
        None

        Return:
        -------
        data_batch: array
            contains a subset of data corresponding to the next batch
        labels_batch: array
            contains a subset of labels corresponding to the next batch

        """
        if self.current_batch_idx == 0:
            self.permutation()
        next_beg = self.current_batch_idx * self.batch_size
        next_end = (self.current_batch_idx + 1) * self.batch_size
        if next_end > self.n_samples:
            next_end = self.n_samples
            self.current_batch_idx = 0
        data_batch = self.data.values[next_beg:next_end][:]
        if onehot is True:
            labels_batch = self.labels_onehot.values[next_beg:next_end][:]
        else:
            labels_batch = self.labels.values[next_beg:next_end][:]
        self.current_batch_idx += 1
        return data_batch, labels_batch

    def permutation(self):
        """
        Run permutations in the dataset to ensure that the different extracted
        sets will contain all type of labels

        Parameters
        ---------

        Return
        ------
        None

        """
        perm = np.random.permutation(self.n_samples)
        self.data = self.data.iloc[perm]
        self.labels = self.labels.iloc[perm]
        self.labels_onehot = self.labels_onehot.iloc[perm]
        self.df_perm = self.df_perm.iloc[perm]

    def random_sampling(self, n_subset):
        """
        Split dataset in n_subset parts and create as many dataset objects
        containing samples of original dataset chosen randomly with replacement

        Parameters
        ----------
        n_subset: unsigned int
            number of sub-sampling datasets to generate

        Returns
        -------
        subset_list: list
            list of n_subset subsampling dataset objects
        """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - Random sampling with replacement ...".format(t))
        subset_list = []
        training_set = self
        subset_size = math.ceil(training_set.n_samples / n_subset)
        # create subsets
        for i in range(n_subset):
            # run a permutation to mix all samples (sampling with replacement)
            self.permutation()
            #  always draw the first samples
            start_idx = 0
            stop_idx = subset_size
            subset = deepcopy(training_set)
            subset.data = subset.data[start_idx:stop_idx][:]
            subset.labels = subset.labels[start_idx:stop_idx][:]
            subset.labels_onehot = subset.labels_onehot[start_idx:stop_idx][:]
            subset.n_samples = stop_idx - start_idx
            subset.true_distribution = subset._get_true_distribution()
            subset.set_batch_size(training_set.batch_size)
            subset_list.append(subset)
            print("\tSubset shape {}".format(subset.data.shape))
        return subset_list

    def _get_true_distribution(self):
        true_distribution = np.array(self.labels.value_counts().sort_index())
        true_distribution = np.int_(true_distribution)
        return true_distribution


class Normalization(object):
    """ Normalization class """

    def __init__(self):
        """
        Normalization object  creator

        Parameters
        ----------

        Return
        ------
        Created object

        """
        self.dict_scalers = {}

    class Scaler(object):
        """ Scaler Class """

        def __init__(self,
                     min_val=None, max_val=None,
                     mean_val=None, std_val=None,
                     q1=None, q3=None,
                     method='min_max_scaling'):
            """
            Scaler object creation

            Parameters
            ----------
            min_val: float
                minimum value of the feature
            max_val: float
                maximum value of the feature
            mean_val: float
                mean value of the feature
            std_val: float
                standard deviation value of the feature
            q1: float
                First quartile value of the feature
            q3: float
                Third quartile value of the feature
            method: str
                supported methods are: 'min_max_scaling', 'z_score_std' or
                'robust_scaling'
            """
            self.min = min_val
            self.max = max_val
            self.mean = mean_val
            self.std = std_val
            self.q1 = q1
            self.q3 = q3
            self.method = method

    def fit(self, df, method='min_max_scaling', per_col_scaler=False):
        """
        Analyze data and create a dictionary of scalers

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize
        method: str
            supported methods are: 'min_max_scaling', 'z_score_std' or '
            robust_scaling'
        per_col_scaler: Boolean
            When True a different scaler is used for each column of the
            DataFrame
            When False, the same scaler is applied to all columns

        Returns
        -------
        None

        """
        # Does df contain multiple columns ?
        if df.size == len(df) or per_col_scaler is True:
            # df contains multiple columns
            lbl_list = df.columns.values
            for lbl in lbl_list:
                try:
                    min_val = float(np.amin(df[lbl]))
                    max_val = float(np.amax(df[lbl]))
                    mean_val = float(np.mean(df[lbl]))
                    std_val = float(np.std(df[lbl]))
                    # TODO Validate/Debug Robust Scaler
                    q1_val = float(np.percentile(df[lbl], 25))
                    q3_val = float(np.percentile(df[lbl], 75))
                except TypeError:
                    raise Exception("[ERROR] TypeError in normalization fit")
                scaler = self.Scaler(min_val=min_val, max_val=max_val,
                                     mean_val=mean_val, std_val=std_val,
                                     q1=q1_val, q3=q3_val,
                                     method=method)
                self.dict_scalers[lbl] = scaler
        else:
            # df contains one single column or scaling is applied
            # independently for each feature/column
            try:
                min_val = float(np.amin(df))
                max_val = float(np.amax(df))
                mean_val = float(np.mean(df))
                std_val = float(np.std(df))
                # TODO Validate/Debug Robust Scaler
                q1_val = float(np.percentile(df, 25))
                q3_val = float(np.percentile(df, 75))
            except TypeError:
                raise Exception("[ERROR] TypeError in normalization fit")
            scaler = self.Scaler(min_val=min_val, max_val=max_val,
                                 mean_val=mean_val, std_val=std_val,
                                 q1=q1_val, q3=q3_val,
                                 method=method)
            self.dict_scalers['OneForAll'] = scaler

    def transform(self, df):
        """
        Calculate normalized DataFrame

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize        df

        Returns
        -------
        normalized_df: DataFrame
            pandas DataFrame containing normalized values

        """
        STATS_ENABLED = False
        # One scaler is required before calling transform
        if len(self.dict_scalers) == 0:
            raise Exception("[ERROR] normalization transform method called"
                            "prior to fit method")

        normalized_df = df.copy()
        if 'OneForAll' in self.dict_scalers:
            sclr = self.dict_scalers['OneForAll']
            if sclr.method == 'min_max_scaling':
                # TODO Saturation case of value >max or <min during fit
                total_range = sclr.max - sclr.min
                if total_range == 0:
                    total_range = 1
                    print("[Warning]: MinMax scaler with min=max")
                normalized_df = (df - sclr.min) / total_range
            elif sclr.method == 'z_score_std':
                normalized_df = (df - sclr.mean) / sclr.std
            elif sclr.method == 'robust_scaling':
                # TODO Validate/Debug Robust Scaler
                iqr = sclr.q3 - sclr.q1
                if iqr == 0:
                    iqr = 1
                    print("[Warning]: robust scaler with q1=q3")
                normalized_df = (df - sclr.q1) / iqr
            else:
                raise Exception("[ERROR] normalization method not "
                                "implemented yet")
        else:
            # Apply parameters to all columns
            lbl_list = df.columns.values
            for lbl in lbl_list:
                sclr = self.dict_scalers[lbl]
                if sclr.method == 'min_max_scaling':
                    # TODO Saturation case of value >max or <min during fit
                    total_range = sclr.max - sclr.min
                    if total_range == 0:
                        total_range = 1
                        print("[Warning]: scaler with min=max for feature: {}".
                              format(lbl))
                    normalized_df[lbl] = (df[lbl] - sclr.min) / total_range
                elif sclr.method == 'z_score_std':
                    normalized_df[lbl] = (df[lbl] - sclr.mean) / sclr.std
                elif sclr.method == 'robust_scaling':
                    # TODO Debug/Validate Robust Scaler
                    iqr = sclr.q3 - sclr.q1
                    if iqr == 0:
                        iqr = 1
                        print("[Warning]: scaler with q1=q3 for feature: {}".
                              format(lbl))
                    normalized_df[lbl] = (df - sclr.q1) / iqr
                else:
                    raise Exception("[ERROR] normalization method not "
                                    "implemented yet")
        if STATS_ENABLED:
            if 'OneForAll' in self.dict_scalers:
                sclr = self.dict_scalers['OneForAll']
                if sclr.method == 'min_max_scaling':
                    bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,
                            0.8, 1]
                elif sclr.method == 'z_score_std':
                    bins = [-1000, -4, -3, -2, -1, 0, 1, 2, 3, 4, 1000]
                else:
                    raise Exception("[ERROR] normalization stats not "
                                    "implemented yet")
                hist, bin_edges = np.histogram(df, bins=bins, density=False)
                t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                template = "[INFO] {} - Histogram - {}: {}"
                print(template.format(t, sclr.method, hist))
            else:
                lbl_list = df.columns.values
                for lbl in lbl_list:
                    sclr = self.dict_scalers[lbl]
                    if sclr.method == 'min_max_scaling':
                        bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,
                                0.8, 1]
                    elif sclr.method == 'z_score_std':
                        bins = [-1000, -4, -3, -2, -1, 0, 1, 2, 3, 4, 1000]
                    else:
                        raise Exception("[ERROR] normalization stats not "
                                        "implemented yet")
                    hist, bin_edges = np.histogram(df[lbl], bins=bins,
                                                   density=False)
                    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    template = "[INFO] {} - Histogram - {} - {}: {}"
                    print(template.format(t, sclr.method, lbl, hist))

        return normalized_df

    def fit_and_transform(self, df, method='min_max_scaling',
                          per_col_scaler=False):
        """
        Analyze data, initialize parameters accordingly and then calculate
        normalized DataFrame

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize
        method: str
            supported methods are: 'min_max_scaling' or 'z_score_std'
        per_col_scaler: Boolean
            When True a different scaler is used for each column of the
            DataFrame
            When False, the same scaler is applied to all columns
        Returns
        -------
        normalized_df: DataFrame
            pandas DataFrame containing normalized values
        """
        self.fit(df, method, per_col_scaler)
        normalized_df = self.transform(df)
        return normalized_df

    @staticmethod
    def clip(df, clip_val_low, clip_val_high):
        """
        Clip values a DataFrame

        Parameters
        ----------
        df: DataFrame
            DataFrame on which the clipping is done
        clip_val_low: float
            Threshold value for the lower bound
        clip_val_high: float
            Threshold value for the upper bound

        Returns
        -------

        """
        clipped_df = df.clip(lower=clip_val_low, upper=clip_val_high)
        return clipped_df

    def quantize(self, df):
        """
        Quantize in 8-bit integer the input DataFrame

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to quantize

        Returns
        -------
        quant_df: DataFrame
            quantized version of df
        """
        if len(self.dict_scalers) == 0:
            raise Exception("[ERROR] quantize method called prior to"
                            "normalization transform method ")

        quant_df = pd.DataFrame()
        if 'OneForAll' in self.dict_scalers:
            # quantization is applied on all features
            min_fp = float(np.amin(df))
            max_fp = float(np.amax(df))
            scale = (max_fp - min_fp) / (127 - (-127))
            zero_point = 127 - (max_fp / scale)
            quant_df = df / scale + zero_point
        else:
            # quantization is applied independently for each feature/column
            lbl_list = df.columns.values
            for lbl in lbl_list:
                min_fp = float(np.amin(df[lbl]))
                max_fp = float(np.amax(df[lbl]))
                scale = (max_fp - min_fp) / (127 - (-127))
                zero_point = 127 - (max_fp / scale)
                quant_df[lbl] = df[lbl] / scale + zero_point
        return quant_df.astype(np.int8)


class HyperParam(object):
    """ Hyperparameter Class """

    def __init__(self, optimizer_name=None,
                 *optim_params,
                 epochs=500, batch_size=32, eval_print_period=0):
        """
        Creator

        Parameters
        ----------
        optimizer_name: str
            Valid strings are: 'GradientDescent', 'Momentum', 'AdaDelta',
            'AdaGrad', 'RMSProp', 'Adam', 'Nadam'
        optim_params:
            Contains variable number of parameters specific to the optimizer
        epochs: int
            Number of epochs used for training
        batch_size: int
            number of instances used in each training step
        eval_print_period: int
            cross-val cost/accuracy displayed every eval_print_period in
            epochs during training
        """
        self.optimizer_name = optimizer_name
        self.optim_params = optim_params
        if optimizer_name is not None and len(optim_params) > 0:
            self.set_optimizer_params()
        else:
            self.optimizer = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_print_period = eval_print_period

    def set_optimizer_params(self):
        """
        Set parameters specific to the optimizer and configure accordingly
        the optimizer

        Parameters
        ----------

        Returns
        -------
        None
        """
        n_params = len(self.optim_params)
        if self.optimizer_name == 'GradientDescent' and n_params == 1:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.optim_params[0],
                momentum=0)
        elif self.optimizer_name == 'Momentum' and n_params == 2:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.optim_params[0],
                momentum=self.optim_params[1])
        elif self.optimizer_name == 'AdaGrad' and n_params == 2:
            self.optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=self.optim_params[0],
                initial_accumulator_value=self.optim_params[1])
        elif self.optimizer_name == 'AdaDelta' and n_params == 2:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.optim_params[0],
                rho=self.optim_params[1])
        elif self.optimizer_name == 'RMSProp' and n_params == 3:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.optim_params[0],
                rho=self.optim_params[1],
                momentum=self.optim_params[2])
        elif self.optimizer_name == 'Adam' and n_params == 3:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.optim_params[0],
                beta_1=self.optim_params[1],
                beta_2=self.optim_params[2])
        elif self.optimizer_name == 'Nadam' and n_params == 3:
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=self.optim_params[0],
                beta_1=self.optim_params[1],
                beta_2=self.optim_params[2])
        else:
            raise Exception("[ERROR] Wrong optimizer or parameters for "
                            "optimizer")


class Layer(object):
    """ Layer Class """

    def __init__(self,
                 name, layer_type, n_nodes, activation_name=None,
                 n_inputs=None, clip_lower=-1, clip_upper=1,
                 activation_scalar=1, l1_reg=0, l2_reg=0, dropout_rate=1,
                 output_bias=None):
        """
        Layer creation

        Parameter
        ---------
        name: str
            layer name
        layer_type: str
            type of layer
            accepted string: 'dense', 'dropout',
            future layers to be supported: 'batchnorm', 'conv2d',
            'conv2dtranspose', 'maxpooling', 'upsampling2d', 'flatten'
            unsupported type will trigger an error
        n_inputs: int
            number of inputs, used only for first layer (enable Model summary)
            default value = None
        activation_name: string
            name of the activation function
            accepted string: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
                             'selu', 'softmax', 'dropout'
            Any other string will be replaced by tensorflow identity function
        l1_reg: float
            L1 regularization factor
        l2_reg: float
            L2 regularization factor
        dropout_rate: float
            probability to drop a node during dropout
            accepted range: 0 < dropout_rate <= 1
            default value = 0 (no dropout)
        output_bias: ndarray
            vector containing one value per class equals to the log of
            probability that a random instance belongs to that class

        Return
        ------
        Created object

        """
        self.name = name
        self.layer_type = layer_type
        if n_inputs is not None:
            self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activation_scalar = activation_scalar
        self.output_bias = output_bias
        self.activation_name = activation_name
        self.lower = clip_lower
        self.upper = clip_upper
        # self.upper = np.random.random((70, 1))

    def create(self):
        """
        Create a tensorflow layer

        Returns
        -------
        outputs: tensor
            outputs of the created layer
        """
        output = None
        if self.output_bias is not None:
            output_bias = tf.keras.initializers.Constant(self.output_bias)
        else:
            output_bias = None
        kernel_init = None
        if self.activation_name == 'relu' or self.activation_name == 'elu':
            #  Kaiming He initialization
            kernel_init = tf.keras.initializers.he_normal()
        elif self.activation_name == 'selu':
            # LeCun initialization
            kernel_init = tf.keras.initializers.lecun_uniform()
        elif self.activation_name == 'tanh' or self.activation_name == \
                'sigmoid':
            # Xavier Glorot initialization
            kernel_init = tf.keras.initializers.glorot_uniform()
        if self.layer_type == 'dense':
            kernel_reg = tf.keras.regularizers.l1_l2(l1=self.l1_reg,
                                                     l2=self.l2_reg)
            output = tf.keras.layers.Dense(units=self.n_nodes,
                                           activation=self.activation_name,
                                           bias_initializer=output_bias,
                                           kernel_initializer=kernel_init,
                                           kernel_regularizer=kernel_reg,
                                           name=self.name)
        # elif self.layer_type == 'clipping':
        #     output = Clipping(self.upper)
        elif self.layer_type == 'dropout':
            output = tf.keras.layers.Dropout(rate=self.dropout_rate,
                                             name=self.name)
        return output


# class Clipping(tf.keras.layers.Layer):
#     """ New keras layer for clipping """
#
#     def __init__(self, alpha):
#         super(Clipping, self).__init__()
#         self.alpha = tf.Variable(alpha, trainable=True, dtype='float32',
#                                  name='alpha')
#
#     def call(self, x_input, trainable=False):
#         x = tf.minimum(tf.maximum(x_input, -self.alpha), self.alpha)
#         return x


# class Clipping_save(tf.keras.layers.Layer):
#     """ New keras layer for clipping """
#
#     def __init__(self, upper):
#         super(Clipping, self).__init__()
#         self.upper = tf.Variable(upper, trainable=True, dtype='float32',
#                                  name='clip_lower')
#         self.alpha = tf.Variable(alpha, trainable=True, dtype='float32',
#                                  name='alpha')
#
#     def call(self, x_input, trainable=False):
#
#         x_lower = tf.scalar_mul(-self.alpha, x_input)
#         x_upper = tf.scalar_mul(self.alpha, x_input)
#         # TODO: correct error in basic version ????
#         x = tf.minimum(tf.maximum(x_input, x_lower), x_upper)
#         #x = tf.scalar_mul(self.upper, tf.tanh(x_input))
#         return x


class Topology(object):
    """ Topology Class """

    def __init__(self):
        """
        Creator function
        """
        self.dict_topo = {}
        self.current = 0

    def add_layer(self, layer):
        """
        Add a layer in the topology description

        Parameters
        ----------
        layer: Layer
            Layer object containing parameters of the layer

        Returns
        -------
        None

        """
        idx = len(self.dict_topo)
        idx += 1
        self.dict_topo[idx] = layer

    def __iter__(self):
        """
        Iterator

        Returns
        -------
        self

        """
        return self

    def __next__(self):
        """
        Next function of Iterator

        Returns
        -------
        layer: Layer
            Next layer of topology

        """
        if self.current >= len(self.dict_topo):
            self.current = 0
            raise StopIteration
        self.current += 1
        return self.dict_topo[self.current]


class NeuralNetworkDescriptor(object):
    """
    Neural Network topology and hyper-parameter descriptions
    Contains also tensorflow descriptor that is automatically filled in during
    graph construction
    """

    def __init__(self, name):
        """
        Creator function
        """
        self.name = name
        self.topology = Topology()
        self.hyper_param = HyperParam()


# noinspection PyCallingNonCallable
class NeuralNetworkModel(tf.keras.Model):
    """
    Model a neural network from keras Model class
    """

    def __init__(self, nn_desc):
        super(NeuralNetworkModel, self).__init__()
        self.n_layers = len(nn_desc.topology.dict_topo)
        self.nn_desc = nn_desc
        self.layer = [None] * self.n_layers
        for idx, layer in self.nn_desc.topology.dict_topo.items():
            self.layer[idx - 1] = layer.create()

    def call(self, x, training=False):
        """
        Function call when neural network model is executed (either for
        training or inference)

        Parameters
        ----------
        x: ndarray
            Array containing input data
        training: tf.bool
            Boolean used to know whether the call is for training or inference
            it should be True for training and False for inference
            Typically used to apply dropout during training

        Returns
        -------
        x: ndarray
            Vector containing outputs of the neural network
        """
        for idx in range(self.n_layers):
            if self.nn_desc.topology.dict_topo[idx+1].layer_type != 'dropout':
                x = self.layer[idx](x)
            else:
                x = self.layer[idx](x, training=tf.dtypes.cast(training,
                                                               dtype=tf.bool))
        return x


# noinspection PyCallingNonCallable
class NeuralNetwork(object):
    """ Neural Network Class """

    def __init__(self, nn_descriptor):
        """
        Creator function
        """
        self.name = nn_descriptor.name
        self.nn_desc = nn_descriptor
        self.model = NeuralNetworkModel(nn_desc=nn_descriptor)
        n_features = self.nn_desc.topology.dict_topo[1].n_inputs
        batch_size = self.nn_desc.hyper_param.batch_size
        self.model.build(input_shape=(batch_size, n_features))
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        self.eval_loss = tf.keras.metrics.Mean(name='test_loss')
        self.eval_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='test_accuracy')
        # Checkpoints are used to save training information
        # TODO use CheckpointManager to limit disk usage for checkpoints
        self.ckpt_dir = "./ckpt_dir/{}".format(self.name)
        self.ckpt_prefix = os.path.join(self.ckpt_dir, "ckpt")
        self.ckpt_handle = tf.train.Checkpoint(
            optimizer=self.nn_desc.hyper_param.optimizer,
            model=self.model,
            step=tf.Variable(0))
        self.best_accuracy = 0.

    def fit(self, *ds, silent_console=False):
        """
        Train neural network

        Parameters
        ----------
        ds: List[int]
            list of Dataset that can contain either one or two dataset
            first element is the training set
            second element is the development set
        silent_console: bool
            When True, no message is put in the console

        Returns
        -------
        loss: float
            Cost value after training
        acc: float
            Accuracy value after training
        cm: ndarray
            confusion matrix
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training
        """
        if len(ds) == 2:
            ds_train = ds[0]
            ds_cv = ds[1]
        elif len(ds) == 1:
            ds_train = ds[0]
            ds_cv = []
        else:
            raise Exception("[ERROR] Wrong parameters in fit()")
        ds_train.set_batch_size(self.nn_desc.hyper_param.batch_size)
        ds_cv.set_batch_size(self.nn_desc.hyper_param.batch_size)
        # Training cycle
        # noinspection PyUnresolvedReferences
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # noinspection PyUnresolvedReferences
        cv_summary_writer = tf.summary.create_file_writer(
            crossval_log_dir)
        n_batch = ds_train.n_batch
        n_epochs = self.nn_desc.hyper_param.epochs
        for epoch in range(int(self.ckpt_handle.step), n_epochs):
            t0 = datetime.now()
            # training using mini-batch
            for train_batch_idx in range(n_batch):
                batch_x, batch_y = ds_train.get_next_batch()
                self._train_step(batch_x, batch_y)
            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            # evaluate on full training set
            tf.summary.trace_on(graph=True, profiler=True)
            self._eval_step(ds_train.data.values[:200000][:],
                            ds_train.labels_onehot.values[:200000][:])
            with cv_summary_writer.as_default():
                tf.summary.trace_export(
                    name="_eval_step",
                    step=0,
                    profiler_outdir=crossval_log_dir)
            train_loss = float(self.eval_loss.result())
            train_accuracy = float(self.eval_accuracy.result())
            self.eval_loss.reset_states()
            self.eval_accuracy.reset_states()
            # training log for tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch+1)
                tf.summary.scalar('accuracy', train_accuracy, step=epoch+1)
            if (epoch + 1) % self.nn_desc.hyper_param.eval_print_period == 0:
                # evaluate on full cross-validation set
                self._eval_step(ds_cv.data.values, ds_cv.labels_onehot.values)
                cv_loss = float(self.eval_loss.result())
                cv_accuracy = float(self.eval_accuracy.result())
                self.eval_loss.reset_states()
                self.eval_accuracy.reset_states()
                # Cross-val log for tensorboard
                with cv_summary_writer.as_default():
                    tf.summary.scalar('loss', cv_loss, step=epoch+1)
                    tf.summary.scalar('accuracy', cv_accuracy, step=epoch+1)
                # Display results with cross-val evaluation
                if silent_console is False:
                    t1 = datetime.now()
                    t = t1.strftime("%Y-%m-%d %H:%M:%S")
                    template = "[INFO] {} - Epoch {}/{} ({:.1f}s), " \
                               "Loss: {:.9f}, Accuracy: {:.9f}, " \
                               "Crossval Loss: {:.9f}, Crossval Accuracy: {:.9f}"
                    print(template.format(t, epoch + 1, n_epochs,
                                          (t1 - t0).total_seconds(),
                                          train_loss,
                                          train_accuracy * 100,
                                          cv_loss,
                                          cv_accuracy * 100))
                # update epochs for tracking in checkpoint
                self.ckpt_handle.step.assign_add(1)
                # save model if results improved
                if cv_accuracy > self.best_accuracy:
                    self.best_accuracy = cv_accuracy
                    # save checkpoint (optimizer state and weights)
                    self.ckpt_handle.save(self.ckpt_prefix)
                    # save saved_model for tensorflow lite usage
                    tf.saved_model.save(self.model,
                                        "./saved_models/{}".format(self.name))
                    if silent_console is False:
                        print("\t\timproved performance --> saving model")
                # print("\t\tclip_upper = {}".format(
                #     self.model.trainable_variables[0]))
                # print("\t\tclip_lower = {}".format(
                #     self.model.trainable_variables[1]))
            else:
                # Display results without cross-val evaluation
                if silent_console is False:
                    t1 = datetime.now()
                    t = t1.strftime("%Y-%m-%d %H:%M:%S")
                    template = "[INFO] {} - Epoch {}/{} ({}s), " \
                               "Loss: {:.9f}, Accuracy: {:.9f}, "
                    print(template.format(t, epoch + 1, n_epochs, t1 - t0,
                                          self.train_loss.result(),
                                          self.train_accuracy.result() * 100))

    def restore(self, ds_crossval):
        """
        load model (weights, epochs) and calculate best cross-vall accuracy)
        from latest checkpoint

        Parameters
        ----------
        ds_crossval: Dataset
            data set used for best accuracy calculation
        Returns
        -------
        None
        """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - Restoring from latest checkpoint ...".format(t))
        self.ckpt_handle.restore(
            tf.train.latest_checkpoint(self.ckpt_dir)).expect_partial()
        self._eval_step(ds_crossval.data.values,
                        ds_crossval.labels_onehot.values)
        self.best_accuracy = float(self.eval_accuracy.result())
        template = "{} model restored after {} epoch(s) - " \
                   "Cross-vall accuracy = {:.9f}"
        print(template.format(self.name, int(self.ckpt_handle.step),
                              self.best_accuracy))

    def get_metrics(self, ds):
        """
        Calculate metrics: accuracy, loss and confusion matrix

        Parameters
        ----------
        ds: Dataset
            dataset containing data and labels for which metrics are calculated

        Returns
        -------
        loss: float
            loss value for the given data and labels
        accuracy: list
            accuracy of the prediction
        pred_: list
            predictions
        cm: ndarray
            confusion matrix
        """
        self._eval_step(ds.data.values, ds.labels_onehot.values)
        loss = float(self.eval_loss.result())
        accuracy = float(self.eval_accuracy.result())
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        outputs = self.model(ds.data.values)
        predictions = tf.math.argmax(outputs, axis=1).numpy()
        cm = confusion_matrix(ds.labels.values, predictions)
        return loss, accuracy, predictions, cm

    # noinspection PyCallingNonCallable
    @tf.function
    def _train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)

            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.nn_desc.hyper_param.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # noinspection PyCallingNonCallable
    @tf.function
    def _eval_step(self, data, labels):
        predictions = self.model(data, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.eval_loss(t_loss)
        self.eval_accuracy(labels, predictions)
