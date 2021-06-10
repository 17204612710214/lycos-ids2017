# -*- coding: utf-8 -*-
#

"""
Filename: create_datasets.py
Date: May 8, 2021 - 14:52
Name: Anonymous
Description:
    -  Extract samples of each traffic type to generate training set,
    cross-validation set and test set

"""
import logging
import pandas as pd
import numpy as np
import os

np.random.seed(2021)

LOG_PATH = "./python_logs/"

LYCOS_PATH = './lycos-ids2017/'
LYCOS_FILELIST = [
    "Monday-WorkingHours.pcap_lycos.csv",
    "Tuesday-WorkingHours.pcap_lycos.csv",
    "Wednesday-WorkingHours.pcap_lycos.csv",
    "Thursday-WorkingHours.pcap_lycos.csv",
    "Friday-WorkingHours.pcap_lycos.csv"
]

ISCX_PATH = './cicids2017/csv_files/'
ISCX_FILELIST = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

ISCX_DATASET_PATH = './datasets/cic-ids2017/'
LYCOS_DATASET_PATH = './datasets/lycos-ids2017/'

FEAT_LIST_ISCX = {
    'flow_id': 0,
    'src_addr': 1,
    'src_port': 2,
    'dst_addr': 3,
    'dst_port': 4,
    'prot': 5,
    'timestamp': 6,
    'flow_duration': 7,
    'tot_fwd_pkt': 8,
    'tot_bwd_pkt': 9,
    'fwd_pkt_len_tot': 10,
    'bwd_pkt_len_tot': 11,
    'fwd_pkt_len_max': 12,
    'fwd_pkt_len_min': 13,
    'fwd_pkt_len_mean': 14,
    'fwd_pkt_len_std': 15,
    'bwd_pkt_len_max': 16,
    'bwd_pkt_len_min': 17,
    'bwd_pkt_len_mean': 18,
    'bwd_pkt_len_std': 19,
    'flow_byte_per_s': 20,
    'flow_pkt_per_s': 21,
    'flow_iat_mean': 22,
    'flow_iat_std': 23,
    'flow_iat_max': 24,
    'flow_iat_min': 25,
    'fwd_iat_tot': 26,
    'fwd_iat_mean': 27,
    'fwd_iat_std': 28,
    'fwd_iat_max': 29,
    'fwd_iat_min': 30,
    'bwd_iat_tot': 31,
    'bwd_iat_mean': 32,
    'bwd_iat_std': 33,
    'bwd_iat_max': 34,
    'bwd_iat_min': 35,
    'fwd_psh_flag': 36,
    'bwd_psh_flag': 37,
    'fwd_urg_flag': 38,
    'bwd_urg_flag': 39,
    'fwd_head_len': 40,
    'bwd_head_len': 41,
    'fwd_pkt_per_s': 42,
    'bwd_pkt_per_s': 43,
    'pkt_len_min': 44,
    'pkt_len_mean': 45,
    'pkt_len_std': 46,
    'pkt_len_max': 47,
    'pkt_len_var': 48,
    'fin_flag_cnt': 49,
    'syn_flag_cnt': 50,
    'rst_flag_cnt': 51,
    'psh_flag_cnt': 52,
    'ack_flag_cnt': 53,
    'urg_flag_cnt': 54,
    'cwe_flag_cnt': 55,
    'ece_flag_cnt': 56,
    'down_up_ratio': 57,
    'avg_pkt_size': 58,
    'avg_fwd_segm_size': 59,
    'avg_bwd_segm_size': 60,
    'fwd_head_len_bis': 61,
    'avg_fwd_byte_per_bulk': 62,
    'avg_fwd_pkt_per_bulk': 63,
    'avg_fwd_bulk_rate': 64,
    'avg_bwd_byte_per_bulk': 65,
    'avg_bwd_pkt_per_bulk': 66,
    'avg_bwd_bulk_rate': 67,
    'subflow_fwd_pkt': 68,
    'subflow_fwd_byte': 69,
    'subflow_bwd_pkt': 70,
    'subflow_bwd_byte': 71,
    'init_win_bytes_fwd': 72,
    'init_win_bytes_bwd': 73,
    'act_data_pkt_fwd': 74,
    'min_seg_size_forward': 75,
    'active_mean': 76,
    'active_std': 77,
    'active_max': 78,
    'active_min': 79,
    'idle_mean': 80,
    'idle_std': 81,
    'idle_max': 82,
    'idle_min': 83,
    'label': 84
}

FEAT_LIST_LYCOS = {
    'flow_id': 0,
    'src_addr': 1,
    'src_port': 2,
    'dst_addr': 3,
    'dst_port': 4,
    'ip_prot': 5,
    'timestamp': 6,
    'flow_duration': 7,
    'down_up_ratio': 8,
    'pkt_len_max': 9,
    'pkt_len_min': 10,
    'pkt_len_mean': 11,
    'pkt_len_var': 12,
    'pkt_len_std': 13,
    'bytes_per_s': 14,
    'pkt_per_s': 15,
    'fwd_pkt_per_s': 16,
    'bwd_pkt_per_s': 17,
    'fwd_pkt_cnt': 18,
    'fwd_pkt_len_tot': 19,
    'fwd_pkt_len_max': 20,
    'fwd_pkt_len_min': 21,
    'fwd_pkt_len_mean': 22,
    'fwd_pkt_len_std': 23,
    'fwd_pkt_hdr_len_tot': 24,
    'fwd_pkt_hdr_len_min': 25,
    'fwd_non_empty_pkt_cnt': 26,
    'bwd_pkt_cnt': 27,
    'bwd_pkt_len_tot': 28,
    'bwd_pkt_len_max': 29,
    'bwd_pkt_len_min': 30,
    'bwd_pkt_len_mean': 31,
    'bwd_pkt_len_std': 32,
    'bwd_pkt_hdr_len_tot': 33,
    'bwd_pkt_hdr_len_min': 34,
    'bwd_non_empty_pkt_cnt': 35,
    'iat_max': 36,
    'iat_min': 37,
    'iat_mean': 38,
    'iat_std': 39,
    'fwd_iat_tot': 40,
    'fwd_iat_max': 41,
    'fwd_iat_min': 42,
    'fwd_iat_mean': 43,
    'fwd_iat_std': 44,
    'bwd_iat_tot': 45,
    'bwd_iat_max': 46,
    'bwd_iat_min': 47,
    'bwd_iat_mean': 48,
    'bwd_iat_std': 49,
    'active_max': 50,
    'active_min': 51,
    'active_mean': 52,
    'active_std': 53,
    'idle_max': 54,
    'idle_min': 55,
    'idle_mean': 56,
    'idle_std': 57,
    'flag_syn': 58,
    'flag_fin': 59,
    'flag_rst': 60,
    'flag_ack': 61,
    'flag_psh': 62,
    'fwd_flag_psh': 63,
    'bwd_flag_psh': 64,
    'flag_urg': 65,
    'fwd_flag_urg': 66,
    'bwd_flag_urg': 67,
    'flag_cwr': 68,
    'flag_ece': 69,
    'fwd_bulk_bytes_mean': 70,
    'fwd_bulk_pkt_mean': 71,
    'fwd_bulk_rate_mean': 72,
    'bwd_bulk_bytes_mean': 73,
    'bwd_bulk_pkt_mean': 74,
    'bwd_bulk_rate_mean': 75,
    'fwd_subflow_bytes_mean': 76,
    'fwd_subflow_pkt_mean': 77,
    'bwd_subflow_bytes_mean': 78,
    'bwd_subflow_pkt_mean': 79,
    'fwd_TCP_init_win_bytes': 80,
    'bwd_TCP_init_win_bytes': 81,
    'label': 82,
}

ISCX_LABEL_DICT = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "PortScan": 9,
    "SSH-Patator": 10,
    "Web Attack \x96 Brute Force": 11,
    "Web Attack \x96 Sql Injection": 12,
    "Web Attack \x96 XSS": 13,
    "Infiltration": 14,
}

LYCOS_LABEL_DICT = {
    "benign": 0,
    "bot": 1,
    "ddos": 2,
    "dos_goldeneye": 3,
    "dos_hulk": 4,
    "dos_slowhttptest": 5,
    "dos_slowloris": 6,
    "ftp_patator": 7,
    "heartbleed": 8,
    "portscan": 9,
    "ssh_patator": 10,
    "webattack_bruteforce": 11,
    "webattack_sql_injection": 12,
    "webattack_xss": 13,
}


def initialize_logger(output_dir):
    """
    Initialize logger so that errors are logged in error.log, all messages
    are logged in all.log printed in console

    Parameters
    ----------
    output_dir: str
        String corresponding to the path of the log file

    Returns
    -------
    None
    """
    pyfile = os.path.splitext(os.path.split(__file__)[1])[0]
    filename = pyfile + "_info.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - "
                                  "%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, filename), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - "
                                  "%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def read_csv(filename):
    """
    load csv file into a DataFrame

    Parameters
    ----------
    filename: str
        file name of the CSV that is loaded

    Returns
    -------
    df: DataFrame
        Pandas DataFrame containing data from CSV
    """
    logging.info("Loading CSV {}".format(filename))
    df = pd.read_csv(filename, encoding="ISO-8859-1", low_memory=False,
                     index_col=False)
    n_rows, n_cols = df.shape
    logging.info("Loaded CSV {} - shape: [{},{}]".format(filename,
                                                         n_rows, n_cols))
    return df


def get_dataframe_ofType(df, traffic_type):
    """
    Extract a traffic type from a pandas data frame

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file
    traffic_type: string
        name corresponding to traffic type

    Return
    ------
    req_df: DataFrame
        Pandas DataFrame containing only the requested traffic type
    """
    req_df = df.loc[df['label'] == traffic_type]
    req_df.reset_index(drop=True, inplace=True)
    return req_df


def remove_empty_lines(df):
    """
    Remove empty lines imported from csv files into Pandas DataFrame as NaN.
    For a fast processing, only FlowID is checked. If NaN, then the line is
    dropped.

    Parameters
    ----------
    df: DataFrame
        Pandas DataFrame to be inspected

    Returns
    -------
    df_clean: DataFrame
        Pandas DataFrame after clean-up
    """
    df.replace([''], np.nan, inplace=True)
    df_clean = df.dropna(subset=['flow_id'], inplace=False)
    n_removed = df.shape[0] - df_clean.shape[0]
    if n_removed != 0:
        logging.info("Empty lines removed: {}".format(n_removed))
    return df_clean


def get_typelist(df):
    """
    Extract traffic type from a pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    traffic_type_list: list
        List of traffic types contained in the DataFrame
    """
    traffic_type_list = df['label'].value_counts().index.tolist()
    return traffic_type_list


def detect_drop_outliers(df):
    """
    Detect and drop NaN rows of a DataFrame

    Parameters
    ----------
    df: DataFrame
        pandas DataFrame containing data

    Returns
    -------
    clean_df: DataFrame
        pandas DataFrame without rows containing NaN
    """

    df.replace(['+Infinity', '-Infinity', 'Infinity'], np.nan,
               inplace=True)
    clean_df = df
    lbl_list = df.columns.values
    lbl_idx, = np.where(lbl_list == 'flow_byte_per_s')
    if lbl_idx.size != 0:
        clean_df['flow_byte_per_s'] = np.array(
            clean_df.iloc[:, lbl_idx]).astype(float)
    lbl_idx, = np.where(lbl_list == 'flow_pkt_per_s')
    if lbl_idx.size != 0:
        clean_df['flow_pkt_per_s'] = np.array(
            clean_df.iloc[:, lbl_idx]).astype(float)
    null_columns = clean_df.columns[clean_df.isna().any()]
    nan_cnt = clean_df[null_columns].isnull().sum()
    if nan_cnt.empty is False:
        for i in range(len(null_columns)):
            logging.info("NaN detected in {}".format(null_columns[i]))
        template = "==> Samples dropped: {}"
        logging.info(template.format(nan_cnt[0]))
        clean_df = clean_df.dropna(axis=0)
        n_rows_prev, n_cols_prev = df.shape
        logging.info("Previous shape: [{}, {}]".format(n_rows_prev,
                                                       n_cols_prev))
        n_rows_new, n_cols_new = clean_df.shape
        logging.info("New shape: [{}, {}]".format(n_rows_new,
                                                  n_cols_new))
    return clean_df


def random_permutation(df_list):
    """
    Run permutations in the dataset

    Parameters
    ---------
    df_list: list
        list of pandas DataFrames, each DataFrames containing one traffic type

    Return
    ------
    reordered_df_list: array
        Resulting array of pandas DataFrames
    """
    df_list_size = len(df_list)
    reordered_df_list = df_list
    for idx in range(df_list_size):
        # Shuffle rows with a given seed to reproduce always same result
        reordered_df_list[idx] = df_list[idx].sample(frac=1, replace=False,
                                                     random_state=0)
    return reordered_df_list


def split_dataset(df_list, randomize, training_percentage,
                  crossval_percentage, test_percentage):
    """
    Split a dataset provided as an array of pandas DataFrames, each DataFrame
    containing one traffic type

    Parameter
    ---------
    df_list: array
        Array of pandas DataFrames
    randomize: boolean
        When True, data lines are permuted randomly before splitting datasets
        (default = False)
    training_percentage: list
        Values (1 per df) between 0 and 100 corresponding to the percentage of
        dataset
        Note that values can be a float
    crossval_percentage: list
        Values (1 per df) between 0 and 100 corresponding to the percentage of
        dataset
        Note that values can be a float
    test_percentage: list
        Values (1 per df) between 0 and 100 corresponding to the percentage of
        dataset
        Note that values can be a float

    Return
    ------
    train_set: DataFrame
        Pandas DataFrame used as training set
    cv_set: DataFrame
        Pandas DataFrame used as cross validation set
    test_set: DataFrame
        Pandas DataFrame used as test set
    """
    logging.info("Splitting dataset ...")
    # Check percentage values
    df_list_size = len(df_list)
    for idx in range(df_list_size):
        if training_percentage[idx] + crossval_percentage[idx] \
                + test_percentage[idx] > 100:
            logging.debug("Sum of percentages > 100")
            exit(-1)
    # Randomize dataset if requested
    if randomize is True:
        df_list = random_permutation(df_list)
    # Declare DataFrame to be returned
    train_set = pd.DataFrame()
    cv_set = pd.DataFrame()
    test_set = pd.DataFrame()
    # Select subset of each dataset except Benign traffic
    key_list = list(LYCOS_LABEL_DICT.keys())
    for idx in range(0, df_list_size):
        n_rows = df_list[idx].shape[0]
        n_training = int(n_rows * training_percentage[idx] / 100)
        n_crossval = int(n_rows * crossval_percentage[idx] / 100)
        template1 = "{} - n_rows: {} - training: {}% - cv: {}%"
        if idx == 14:
            traffic_type = 'Infiltration'
        else:
            traffic_type = key_list[idx]
        logging.info(template1.format(traffic_type, n_rows,
                                      training_percentage[idx],
                                      crossval_percentage[idx]))
        template2 = "{} - training instances: {} - crossval/test instances: {}"
        logging.info(template2.format(traffic_type, n_training, n_crossval))
        n_test = int(n_rows * test_percentage[idx] / 100)
        training_end = n_training
        crossval_end = training_end + n_crossval
        test_end = crossval_end + n_test
        train_set = train_set.append(df_list[idx][:training_end])
        cv_set = cv_set.append(df_list[idx][training_end:crossval_end])
        test_set = test_set.append(df_list[idx][crossval_end:test_end])
    # Shuffle datasets
    train_set = train_set.sample(frac=1)
    cv_set = cv_set.sample(frac=1)
    test_set = test_set.sample(frac=1)
    # Display size of each DataFrame
    logging.info("Training set shape: {}".format(train_set.shape))
    logging.info("Cross-val set shape: {}".format(cv_set.shape))
    logging.info("Test set shape: {}".format(test_set.shape))
    # return resulting DataFrames
    return train_set, cv_set, test_set


def detect_non_informative_features(df_train, df_cv, df_test):
    """
    Detection of features that do not carry any information

    Parameters
    ----------
    df_train: DataFrame
        Contains training instances
    df_cv: DataFrame
        Contains crossval instances
    df_test: DataFrame
        Contains test instances

    Returns
    -------
    feature_list: list
        Contains features that can be dropped
    """
    feature_set = df_train.columns
    feature_list = []
    for i, feat in enumerate(feature_set):
        val_min_cv = df_cv[feature_set[i]].min()
        val_max_cv = df_cv[feature_set[i]].max()
        if val_min_cv == val_max_cv:
            val_min_test = df_test[feature_set[i]].min()
            val_max_test = df_test[feature_set[i]].max()
            if val_min_test == val_max_test:
                val_min_train = df_train[feature_set[i]].min()
                val_max_train = df_train[feature_set[i]].max()
                if val_min_train == val_max_train:
                    feature_list.append(feat)
                    template = "Feature {} can be dropped - min = max = {}"
                    logging.info(template.format(feat, val_min_train))
    return feature_list


def main():
    """
    Main program

    """
    # create output directories if they don't exist yet
    if not os.path.exists(ISCX_DATASET_PATH):
        os.makedirs(ISCX_DATASET_PATH)
    if not os.path.exists(LYCOS_DATASET_PATH):
        os.makedirs(LYCOS_DATASET_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    initialize_logger(LOG_PATH)

    df_list = [
        pd.DataFrame(),  # benign
        pd.DataFrame(),  # bot
        pd.DataFrame(),  # ddos
        pd.DataFrame(),  # dos_goldeneye
        pd.DataFrame(),  # dos_hulk
        pd.DataFrame(),  # dos_slowhttptest
        pd.DataFrame(),  # dos_slowloris
        pd.DataFrame(),  # ftp_patator
        pd.DataFrame(),  # heartbleed
        pd.DataFrame(),  # portscan
        pd.DataFrame(),  # ssh_patator
        pd.DataFrame(),  # webattack_bruteforce
        pd.DataFrame(),  # webattack_xss
        pd.DataFrame(),  # webattack_sqlinjection
        pd.DataFrame()  # infiltration
    ]

    # CIC-IDS2017 original dataset
    # loop over each file
    for filename in ISCX_FILELIST:
        # Load one file as a data frame
        df = read_csv(ISCX_PATH + filename)
        # convert col labels
        col_names = list(FEAT_LIST_ISCX.keys())
        df.columns = col_names

        typelist = get_typelist(df)
        df = remove_empty_lines(df)
        for traffic_type in typelist:
            logging.info("Extracting {} ...".format(traffic_type))
            idx = ISCX_LABEL_DICT[traffic_type]
            # idx = string2index(traffic_type)
            df_list[idx] = df_list[idx].append(
                get_dataframe_ofType(df, traffic_type), sort=False)
            df_list[idx] = detect_drop_outliers(df_list[idx])

    # align ISCX labels to Lycos labels
    key_list = list(LYCOS_LABEL_DICT.keys())
    for key, val in ISCX_LABEL_DICT.items():
        if key != "Infiltration":
            df_list[val]['label'] = key_list[val]
        # template = "{}: {} instances"
        # logging.info(template.format(key, df_list[val].shape[0]))

    # split dataset for training, cross validation and test
    # each value is a percentage of each traffic type so that:
    # - training percentage is 50% (+25% for crossval, +25% for test)
    # - attacks represent 50% of the dataset
    # note that infiltration is removed because it is absent of lycos-ids2017
    training_percentage = [
        9.6941,   # benign
        18.6674,  # bot
        37.3679,  # ddos
        32.8573,  # dos_goldeneye
        34.544,   # dos_hulk
        44.2445,  # dos_slowhttptest
        48.9476,  # dos_slowloris
        25.2079,  # ftp_patator
        45.4546,  # heartbleed
        50.00,    # portscan
        25.0806,  # ssh_patator
        45.1228,  # webattack_bruteforce
        28.5715,  # webattack_sql_injection
        50.0000,  # webattack_xss
        0         # Infiltration
    ]

    crossval_percentage = [x / 2 for x in training_percentage]
    test_percentage = crossval_percentage
    (df_train,
     df_cv,
     df_test) = split_dataset(df_list, randomize=True,
                              training_percentage=training_percentage,
                              crossval_percentage=crossval_percentage,
                              test_percentage=test_percentage)
    # identify useless features
    detect_non_informative_features(df_train, df_cv, df_test)
    # write in parquet file
    logging.info("Writing ISCX Parquet files ...")
    df_train.to_parquet(ISCX_DATASET_PATH + "train_set.parquet")
    df_cv.to_parquet(ISCX_DATASET_PATH + "crossval_set.parquet")
    df_test.to_parquet(ISCX_DATASET_PATH + "test_set.parquet")

    # LYCOS-IDS2017 dataset
    df_list = [
        pd.DataFrame(),  # benign
        pd.DataFrame(),  # bot
        pd.DataFrame(),  # ddos
        pd.DataFrame(),  # dos_goldeneye
        pd.DataFrame(),  # dos_hulk
        pd.DataFrame(),  # dos_slowhttptest
        pd.DataFrame(),  # dos_slowloris
        pd.DataFrame(),  # ftp_patator
        pd.DataFrame(),  # heartbleed
        pd.DataFrame(),  # portscan
        pd.DataFrame(),  # ssh_patator
        pd.DataFrame(),  # webattack_bruteforce
        pd.DataFrame(),  # webattack_xss
        pd.DataFrame(),  # webattack_sqlinjection
    ]
    # loop over each file
    for filename in LYCOS_FILELIST:
        # Load one file as a data frame
        df = read_csv(LYCOS_PATH + filename)

        typelist = get_typelist(df)
        df = remove_empty_lines(df)
        for traffic_type in typelist:
            logging.info("Extracting {} ...".format(traffic_type))
            idx = LYCOS_LABEL_DICT[traffic_type]
            df_list[idx] = df_list[idx].append(
                get_dataframe_ofType(df, traffic_type), sort=False)
            df_list[idx] = detect_drop_outliers(df_list[idx])

    for key, val in LYCOS_LABEL_DICT.items():
        template = "{}: {} instances"
        logging.info(template.format(key, df_list[val].shape[0]))

    # split dataset for training, cross validation and test
    # each value is a percentage of each traffic type so that:
    # - training percentage is 50% (+25% for crossval, +25% for test)
    # - attacks represent 50% of the dataset
    # note that infiltration is removed because it is absent of lycos-ids2017
    training_percentage = [
        15.78565,  # benign
        50.00000,  # bot
        50.00000,  # ddos
        50.00000,  # dos_goldeneye
        50.00000,  # dos_hulk
        50.00000,  # dos_slowhttptest
        50.00000,  # dos_slowloris
        50.00000,  # ftp_patator
        50.00000,  # heartbleed
        49.63280,  # portscan
        50.00000,  # ssh_patator
        50.00000,  # webattack_bruteforce
        50.00000,  # webattack_sql_injection
        49.31930,  # webattack_xss
    ]

    crossval_percentage = [x / 2 for x in training_percentage]
    test_percentage = crossval_percentage
    (df_train,
     df_cv,
     df_test) = split_dataset(df_list, randomize=True,
                              training_percentage=training_percentage,
                              crossval_percentage=crossval_percentage,
                              test_percentage=test_percentage)
    # identify useless features
    detect_non_informative_features(df_train, df_cv, df_test)
    # write in parquet file
    logging.info("Writing Lycos Parquet files ...")
    df_train.to_parquet(LYCOS_DATASET_PATH + "train_set.parquet")
    df_cv.to_parquet(LYCOS_DATASET_PATH + "crossval_set.parquet")
    df_test.to_parquet(LYCOS_DATASET_PATH + "test_set.parquet")

    logging.info("----Program finished----")


if __name__ == "__main__":
    main()
