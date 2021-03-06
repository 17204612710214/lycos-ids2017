#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Filename: labelling_ids2017.py
Date: May 1, 2021 - 13:51
Name: Anonymous
Description:
    - Labelling flows of CSV files generated by LycoSTand
"""

import pandas as pd
import logging
import os
import glob
import multiprocessing as mp
import time
import numpy as np

# CONSTANTS
LOG_PATH = "./python_logs/"
INPUT_PATH_LYCOS = "./pcap_lycos/"
OUTPUT_PATH_LABELLED = "./lycos-ids2017/"
EXT = ".csv"

LYCOS_FILELIST = [
    "Monday-WorkingHours.pcap_lycos",
    "Tuesday-WorkingHours.pcap_lycos",
    "Wednesday-WorkingHours.pcap_lycos",
    "Thursday-WorkingHours.pcap_lycos",
    "Friday-WorkingHours.pcap_lycos"
    ]


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
    # now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = pyfile + "_info_" + now + ".log"
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
    process_name = mp.current_process().name
    template = "{} - Loading CSV {}"
    logging.info(template.format(process_name, filename))
    df = pd.read_csv(filename, encoding="ISO-8859-1", low_memory=False,
                     index_col=False)
    template = "{} - Loaded CSV {}"
    logging.info(template.format(process_name, filename))
    return df


def write_csv(df, filename):
    """
    Write content of DataFrame into a CSV file
    Parameters
    ----------
    df: DataFrame
        DataFrame containing data to be written in CSV
    filename: str
        File name of the CSV to write

    Returns
    -------
    None
    """
    process_name = mp.current_process().name
    template = "{} - Writing CSV {}"
    logging.info(template.format(process_name, filename))
    df.to_csv(filename, index=False)
    template = "{} - Written CSV {}"
    logging.info(template.format(process_name, filename))


def sort_by_epoch(df):
    """
    Sort a Pandas DataFrame by the column containing Epoch Time in an
    ascending order.
    The column name is passed as a parameter because it differs depending on
    the source.
    csv file being original or generated.

    Parameters
    ----------
    df: DataFrame
        Pandas DataFrame to be sorted

    Returns
    -------
    df_out: DataFrame
        sorted Pandas DataFrame
    """
    process_name = mp.current_process().name
    template = "{} - Sorting by epoch time"
    logging.info(template.format(process_name))
    df_out = df.sort_values(by=['timestamp',
                                'src_port', 'dst_port',
                                'src_addr', 'dst_addr'],
                            ascending=[True,
                                       True, True,
                                       True, True],
                            inplace=False)
    df_out.reset_index(drop=True, inplace=True)
    template = "{} - Sorted all lines"
    logging.info(template.format(process_name))
    return df_out


def labelling_mon(df_lycos):
    """
    Set labels for Monday

    Parameters
    ----------
    df_lycos: DataFrame
        DataFrame generated by CicFlowMeter for Monday

    Returns
    -------
    df_out: DataFrame
        DataFrame with updated labels
    """
    process_name = mp.current_process().name

    # all flows are benign on Monday
    df_lycos.loc[:, 'label'] = 'benign'

    # count instances of each traffic in both iscx and cfm
    lycos_stat = df_lycos['label'].value_counts()
    template = "{} - Lycos {}: {}"
    for i in range(lycos_stat.shape[0]):
        logging.info(template.format(process_name,
                                     lycos_stat.index[i], lycos_stat.iloc[i]))

    return df_lycos


# def labelling_tue(df_lycos, df_iscx):
def labelling_tue(df_lycos):
    """
    Set labels by mapping iscx flows with CicFlowMeter flows for Wednesday

    Parameters
    ----------
    df_lycos: DataFrame
        DataFrame generated by CicFlowMeter for Tuesday

    Returns
    -------
    df_out: DataFrame
        DataFrame with updated labels
    """
    process_name = mp.current_process().name

    # processing ftp patator attacks
    ftp_period_lycos = ((df_lycos['timestamp'] >= 1499170620000000) &
                        (df_lycos['timestamp'] <= 1499175000000000))
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                 (df_lycos['dst_addr'] == "192.168.10.50")
    subset_lycos = subset_lycos & ftp_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'ftp_patator'

    # processing ssh patator attacks
    ssh_period_lycos = ((df_lycos['timestamp'] >= 1499188140000000) &
                        (df_lycos['timestamp'] <= 1499191860000000))
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                 (df_lycos['dst_addr'] == "192.168.10.50")
    subset_lycos = subset_lycos & ssh_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'ssh_patator'

    # remaining flows are benign
    df_lycos.loc[df_lycos['label'] == "NeedLabel", 'label'] = "benign"

    # count instances of each traffic in both iscx and cfm
    lycos_stat = df_lycos['label'].value_counts()
    template = "{} - Lycos {}: {}"
    for i in range(lycos_stat.shape[0]):
        logging.info(template.format(process_name,
                                     lycos_stat.index[i], lycos_stat.iloc[i]))

    return df_lycos


# def labelling_wed(df_lycos, df_iscx):
def labelling_wed(df_lycos):
    """
    Set labels by mapping iscx flows with CicFlowMeter flows for Wednesday

    Parameters
    ----------
    df_lycos: DataFrame
        DataFrame generated by CicFlowMeter for Tuesday

    Returns
    -------
    df_out: DataFrame
        DataFrame with updated labels
    """
    process_name = mp.current_process().name
    # processing heartbleed attacks
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                   (df_lycos['src_port'] == 45022) & \
                   (df_lycos['dst_addr'] == "192.168.10.51") & \
                   (df_lycos['dst_port'] == 444)
    df_lycos.loc[subset_lycos, 'label'] = 'heartbleed'

    # processing DoS Slowloris
    slowloris_period_lycos = (((df_lycos['timestamp'] >= 1499256060000000) &
                               (df_lycos['timestamp'] <= 1499260260000000)) |
                              ((df_lycos['timestamp'] >= 1499275440000000) &
                               (df_lycos['timestamp'] <= 1499275500000000)))
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                   (df_lycos['dst_addr'] == "192.168.10.50") & \
                   (df_lycos['dst_port'] == 80)
    subset_lycos = subset_lycos & slowloris_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'dos_slowloris'

    # processing DoS SlowHTTPtest attacks
    slowhttptest_period_lycos = (df_lycos['timestamp'] >= 1499260500000000) & \
                                (df_lycos['timestamp'] <= 1499261820000000)
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                   (df_lycos['dst_addr'] == "192.168.10.50") & \
                   (df_lycos['dst_port'] == 80)
    subset_lycos = subset_lycos & slowhttptest_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'dos_slowhttptest'

    # processing DoS Hulk attacks
    hulk_period_lycos = (df_lycos['timestamp'] >= 1499262180000000) & \
                        (df_lycos['timestamp'] <= 1499263620000000)
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                   (df_lycos['dst_addr'] == "192.168.10.50") & \
                   (df_lycos['dst_port'] == 80)
    subset_lycos = subset_lycos & hulk_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'dos_hulk'

    # processing DoS GoldenEye attacks
    goldeneye_period_lycos = (df_lycos['timestamp'] >= 1499263800000000) & \
                             (df_lycos['timestamp'] <= 1499264340000000)
    subset_lycos = (df_lycos['src_addr'] == "172.16.0.1") & \
                   (df_lycos['dst_addr'] == "192.168.10.50") & \
                   (df_lycos['dst_port'] == 80)
    subset_lycos = subset_lycos & goldeneye_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'dos_goldeneye'

    # remaining flows are benign
    df_lycos.loc[df_lycos['label'] == "NeedLabel", 'label'] = "benign"

    # count instances of each traffic in both iscx and Lycos
    Lycos_stat = df_lycos['label'].value_counts()
    template = "{} - Lycos {}: {}"
    for i in range(Lycos_stat.shape[0]):
        logging.info(template.format(process_name,
                                     Lycos_stat.index[i], Lycos_stat.iloc[i]))

    return df_lycos


def labelling_thu(df_lycos):
    """
    Set labels by mapping iscx flows with CicFlowMeter flows for Wednesday

    Parameters
    ----------
    df_lycos: DataFrame
        DataFrame generated by CicFlowMeter for Tuesday

    Returns
    -------
    df_out: DataFrame
        DataFrame with updated labels
    """
    process_name = mp.current_process().name

    # processing web attacks - brute force
    web_bf_period_lycos = ((df_lycos['timestamp'] >= 1499343300000000) &
                           (df_lycos['timestamp'] <= 1499346000000000))
    subset_lycos = ((df_lycos['src_addr'] == "172.16.0.1") &
                    (df_lycos['dst_addr'] == "192.168.10.50") &
                    (df_lycos['ip_prot'] == 6))
    subset_lycos = subset_lycos & web_bf_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'webattack_bruteforce'

    # processing web attacks - xss
    web_xss_period_lycos = ((df_lycos['timestamp'] >= 1499346935000000) &
                            (df_lycos['timestamp'] <= 1499348100000000))
    subset_lycos = ((df_lycos['src_addr'] == "172.16.0.1") &
                    (df_lycos['dst_addr'] == "192.168.10.50") &
                    (df_lycos['ip_prot'] == 6))
    subset_lycos = subset_lycos & web_xss_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'webattack_xss'

    # processing web attacks - sql injection
    web_sql_period_lycos = ((df_lycos['timestamp'] >= 1499348400000000) &
                            (df_lycos['timestamp'] <= 1499348576000000))
    subset_lycos = ((df_lycos['src_addr'] == "172.16.0.1") &
                    (df_lycos['dst_addr'] == "192.168.10.50") &
                    (df_lycos['ip_prot'] == 6))
    subset_lycos = subset_lycos & web_sql_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'webattack_sql_injection'

    # remove all flow of Thursday afternoon
    drop_period_lycos = (df_lycos['timestamp'] >= 1499353200000000)
    df_lycos.drop(df_lycos[drop_period_lycos].index, axis=0, inplace=True)
    # remaining flows are benign
    df_lycos.loc[df_lycos['label'] == "NeedLabel", 'label'] = "benign"

    # count instances of each traffic in both iscx and Lycos
    Lycos_stat = df_lycos['label'].value_counts()
    template = "{} - Lycos {}: {}"
    for i in range(Lycos_stat.shape[0]):
        logging.info(template.format(process_name,
                                     Lycos_stat.index[i], Lycos_stat.iloc[i]))

    return df_lycos


def labelling_fri(df_lycos):
    """
    Set labels by mapping iscx flows with CicFlowMeter flows for Wednesday

    Parameters
    ----------
    df_lycos: DataFrame
        DataFrame generated by CicFlowMeter for Tuesday

    Returns
    -------
    df_out: DataFrame
        DataFrame with updated labels
    """
    process_name = mp.current_process().name

    # drop dst_addr = "205.174.165.73" after 1499436122: portscan attack
    # wrongly labelled
    drop_wrong_labels = ((df_lycos['dst_addr'] == "205.174.165.73") &
                         (df_lycos['timestamp'] > 1499436193000000))
    df_lycos.drop(df_lycos[drop_wrong_labels].index, axis=0, inplace=True)

    # processing bot attacks
    bot_period_lycos = ((df_lycos['timestamp'] >= 1499430840000000) &
                        (df_lycos['timestamp'] <= 1499436122000000))
    subset_lycos = (df_lycos['dst_addr'] == "205.174.165.73") & bot_period_lycos
    df_lycos.loc[subset_lycos, 'label'] = 'bot'

    # drop flows with ip addr 52.6.13.28 and 52.7.235.158 (labelled as Bot
    # in ISCX but not mentioned on website, seems normal - might be
    # communication with Botnet C&C?)
    drop_subset = ((df_lycos['dst_addr'] == "52.6.13.28") |
                   (df_lycos['dst_addr'] == "52.7.235.158"))
    df_lycos.drop(df_lycos[drop_subset].index, axis=0, inplace=True)

    # processing DDoS attacks
    ddos_period_lycos = ((df_lycos['timestamp'] >= 1499453791000000) &
                         (df_lycos['timestamp'] <= 1499454973000000))
    subset_lycos = (((df_lycos['src_addr'] == "172.16.0.1") &
                     (df_lycos['dst_addr'] == "192.168.10.50") &
                     (df_lycos['ip_prot'] == 6)) &
                     ddos_period_lycos)
    df_lycos.loc[subset_lycos, 'label'] = 'ddos'

    # processing PortScan attacks                  1499449976826698
    portscan_period_lycos = ((df_lycos['timestamp'] >= 1499443530000000) &
                             (df_lycos['timestamp'] <= 1499451842000000))
    subset_lycos = (((df_lycos['src_addr'] == "172.16.0.1") &
                     (df_lycos['dst_addr'] == "192.168.10.50") &
                    (df_lycos['ip_prot'] == 6)) &
                    portscan_period_lycos)
    df_lycos.loc[subset_lycos, 'label'] = 'portscan'

    # remaining flows are benign
    df_lycos.loc[df_lycos['label'] == "NeedLabel", 'label'] = "benign"

    # count instances of each traffic in both iscx and Lycos
    lycos_stat = df_lycos['label'].value_counts()
    template = "{} - Lycos {}: {}"
    for i in range(lycos_stat.shape[0]):
        logging.info(template.format(process_name,
                                     lycos_stat.index[i], lycos_stat.iloc[i]))

    return df_lycos


def thread_func(filename):
    """
    Function that will process the labelling of one file

    Parameters
    ----------
    filename: str
        File name of the generated CSV file to label (written without the extension)

    Returns
    -------
    None
    """
    mp.current_process().name = filename[:3] + "_thread"
    process_name = mp.current_process().name
    template = "{} - Labelling the flows of the generated files"
    logging.info(template.format(process_name))
    df_lycos = read_csv(INPUT_PATH_LYCOS + filename + EXT)
    df_lycos = sort_by_epoch(df_lycos)
    if filename == "Monday-WorkingHours.pcap_lycos":
        df_lycos = labelling_mon(df_lycos)
    elif filename == "Tuesday-WorkingHours.pcap_lycos":
        df_lycos = labelling_tue(df_lycos)
    elif filename == "Wednesday-WorkingHours.pcap_lycos":
        df_lycos = labelling_wed(df_lycos)
    elif filename == "Thursday-WorkingHours.pcap_lycos":
        df_lycos = labelling_thu(df_lycos)
    elif filename == "Friday-WorkingHours.pcap_lycos":
        df_lycos = labelling_fri(df_lycos)
    else:
        raise Exception("filename issue")
    write_csv(df_lycos, OUTPUT_PATH_LABELLED + filename + EXT)


def main():
    # create output directories if they don't exist yet
    if not os.path.exists(OUTPUT_PATH_LABELLED):
        os.makedirs(OUTPUT_PATH_LABELLED)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    initialize_logger(LOG_PATH)
    # show # of CPUs and files
    n_cpu = mp.cpu_count()
    n_files = len(LYCOS_FILELIST)
    template = "Number of CPUs: {} - Number of files: {}"
    logging.info(template.format(n_cpu, n_files))
    time.sleep(1)
    # select number of thread
    n_threads = 0
    max_thread = min(n_cpu, n_files)
    # DEBUG = True
    DEBUG = False
    if DEBUG is True:
        thread_func(LYCOS_FILELIST[2])
    else:
        loop_exit = False
        while not loop_exit:
            template = "Enter # of processing thread (1-{}): ".format(
                max_thread)
            n_threads = int(input(template))
            if 0 < n_threads <= max_thread:
                loop_exit = True
            else:
                print("!!! Correct values: from {} to {} !!!".format(0,
                                                                     max_thread))
        # process all files using the selected number of threads
        pool = mp.Pool(n_threads)
        pool.map(thread_func, LYCOS_FILELIST)
    logging.info("----Program finished----")


if __name__ == '__main__':
    main()
