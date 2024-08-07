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
@File - converts textual data file to CSV format.
"""

import os
import re
import shutil
import tarfile
import requests

import numpy as np
import pandas as pd

from pathlib import Path
from zipfile import ZipFile

from src.utility import load_config

# Load config
config = load_config('config.yaml')


def download_file(url, local_path):
    # Ensure local_path is a Path object
    local_path = Path(local_path)

    # Check if file exists locally
    if not local_path.exists():
        print('='*60)
        print("File not found locally. Downloading...")
        response = requests.get(url, stream=True)
        with local_path.open('wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download completed.")
    else:
        print("File already exists locally.")


def extract_file(local_path, extract_to):
    # Ensure local_path and extract_to are Path objects
    local_path = Path(local_path)
    extract_to = Path(extract_to)

    if local_path.suffixes == ['.tar', '.gz']:
        print(f"Extracting {local_path} to {extract_to}...")
        with tarfile.open(local_path, "r:gz") as tar_ref:
            tar_ref.extractall(path=extract_to)
            print(f"Extracted files to {extract_to}")
    elif local_path.suffix == '.zip':
        print(f"Extracting {local_path} to {extract_to}...")
        with ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to)
            print(f"Extracted files to {extract_to}")


def show_file_size(file):
    return round(os.stat(file).st_size / (1024 * 1024), 2)


def get_files(DATA_DIR):
    '''
    returns the list of all data files in the given data directory
    '''
    labeled_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(DATA_DIR):
        for file in f:
            if file.endswith(".labeled"):
                file_name = os.path.join(r, file)
                labeled_files.append(file_name)
                print(f"File: {file_name.rsplit('IoTScenarios', 2)[-1]} \
                    -> Size: {show_file_size(file_name)}")
    return labeled_files


def line2row(str_line):
    '''
    converts a line of the file to a dataframe row
    '''
    row = None
    if len(str_line) > 0:
        if str_line[0] != '#' or str_line[0] != ' ':
            row = str_line.replace('#', '').replace('.', '_').strip()
            row = re.split(r'\t|\s+', row)
            # row = re.split(r'\t', row)
    return row


def get_columns(file):
    '''define the columns for the dataframe
    '''
    cols = None
    with open(file, 'r') as f:
        head_rows = [next(f) for x in range(20)]
    if head_rows:
        for i in range(len(head_rows)):
            if head_rows[i][0] != '#' and head_rows[i][0] != ' ':
                # -2 because the header line is located two lines \
                # above than first data row line.
                cols = line2row(head_rows[i-2])
                break
    else:
        print('No header found!')
    return list(cols)


def save_dict2df(data, cols, csv_file=None):
    '''
    save the dataframe as a csv file
    param: data: can be either dict_list or dataframe
    cols: list of columns
    csv_file: name of the file to save the final dataframe.
    return: dataframe df if csv_file param is None/disabled which is for chunk processing.
    '''
    df = pd.DataFrame()
    if config['preprocess']['save_csv'] == 'full':
        # correction of the last three columns
        df = pd.DataFrame.from_dict(data)
        last_col = df.columns[-1]
        last_three_cols = ['tunnel_parents', 'label', 'detailed-label']
        df[last_three_cols] = df[last_col].str.split(r'\s+', expand=True)
        df.drop(last_col, axis=1, inplace=True)
        df.columns = cols
        assert len(df.columns) == len(cols), \
            f'Error on renaming the columns! len(df.columns) should be equal to len(cols)'
        df = df.drop(columns=['ts', 'uid', 'id_orig_h',
                     'id_orig_p', 'id_resp_h', 'id_resp_p'], axis=1)
    else:
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame.from_records(data, columns=cols)
            df = df.drop(columns=['ts', 'uid', 'id_orig_h',
                         'id_orig_p', 'id_resp_h', 'id_resp_p'], axis=1)

    df = df.drop_duplicates()
    # print('Head of the processed data file: \n', df.head(2))
    print(f'Shape of the dataframe: {df.shape}')
    if csv_file:
        df.to_csv(csv_file, index=False)
        size = show_file_size(csv_file)
        print(f'Saved CSV file at: {csv_file} (Size: {size} MB)')
        print('-'*60)
    else:
        return df


def file2rows(config, labeled_files, cols, chunk_size=1000000):
    chunk_dir = config['data_dir'] + 'raw/chunks-filter/'
    if os.path.exists(chunk_dir):
        # remove directory if exists
        shutil.rmtree(chunk_dir, ignore_errors=True)

    # create directory if not exists
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)

    for file in labeled_files:
        fc = 1
        file_size = round(os.stat(file).st_size / (1024 * 1024), 2)
        print('\n' + '*'*100)
        print(f'\n{fc}. Processing file: {file}\n File Size is {file_size} MB')
        print('-'*60)

        with open(file, 'r') as fp:
            fpart = file.rsplit("IoTScenarios", 2)[-1].split('/')[1]
            csv_file = chunk_dir + fpart + '.csv'
            max_rows = config['debug_rows'] if config['debug'] else None
            dict_list = []

            if config['preprocess']['save_csv'] == 'full':
                if max_rows:
                    np_ary = np.loadtxt(file,
                                        delimiter='\t',
                                        comments='#',  # default is '#'
                                        dtype=str,
                                        max_rows=config['debug_rows'],
                                        unpack=False  # return a tuple of ndarray
                                        )
                else:
                    np_ary = np.loadtxt(file,
                                        delimiter='\t',
                                        comments='#',  # default is '#'
                                        dtype=str,
                                        unpack=False  # return a tuple of ndarray
                                        )
                if len(np_ary) == 0:
                    print('No data found in the file!')
                    continue
                else:
                    df_file = pd.DataFrame.from_records(np_ary)
                    save_dict2df(np_ary, cols, csv_file)
                    dict_list = []

            elif config['preprocess']['save_csv'] == 'chunk':
                count_rows = 1
                df_file = pd.DataFrame()
                for line in fp:
                    row = line2row(line)
                    if row and len(row) == len(cols):
                        # print(f'row lenght:{len(row)} and len(cols): {len(cols)}')
                        dict_list.append(row)  # df.append(row) is slower
                    else:
                        if len(row) > 10 and row[0] == 'types' and row[0] == 'fields':
                            print(f'\nError on appending the row, either by len \
                                not equals to cols len or None value:   \
                                    len: {len(row)}\n  row: {row}')
                    # saving chunks of a large file into different files
                    if count_rows % chunk_size == 0:
                        # save_dict2df(dict_list, cols, csv_file.split('.csv')[0] \
                        #     + '_' + str(count_rows) + '.csv')
                        df_chunk = save_dict2df(
                            data=dict_list, cols=cols, csv_file=None)
                        df_file = pd.concat(
                            [df_file, df_chunk], ignore_index=True)
                        # may have some duplicates after merging each chunk
                        df_file = df_file.drop_duplicates()
                        dict_list = []  # reset it for next chunk
                    count_rows += 1

                # saving small or last remaining chunk of the file
                if count_rows % chunk_size > 0:
                    if count_rows < chunk_size:
                        save_dict2df(dict_list, cols, csv_file)
                    else:
                        # save_dict2df(dict_list, cols, csv_file.split('.csv')[0] \
                        #         + '_' + str(count_rows) + '.csv')
                        df_chunk = save_dict2df(
                            data=dict_list, cols=cols, csv_file=None)
                        df_file = pd.concat(
                            [df_file, df_chunk], ignore_index=True)
                        print(
                            'Shape of the file before filering on merged chunks:', df_file.shape)
                        save_dict2df(df_file, cols, csv_file)

                    dict_list = []  # reset the list
            else:
                print(
                    'The type of csv saving method is not known. It should be chunk/full')
        fc += 1

    print('-'*60)
    print('All textual files have been converted to csv files!')
    print('-'*60)


if __name__ == '__main__':

    # iot23_dir = config['preprocess']['iot23_dir']

    # Download and extract the dataset
    iot23_url = config['preprocess']['iot23_url']
    raw_iot23_zip = "data/raw/iot23_datasets_small.tar.gz"
    iot23_dir = Path(raw_iot23_zip).parent / 'iot23_datasets_small'

    # check raw_iot23_zip file exists or not
    if not Path(raw_iot23_zip).exists():
        print(f"File {raw_iot23_zip} not found!")

        print(f"Downloading and extracting the dataset to {iot23_dir}...")
        # Ensure the directory exists
        iot23_dir.mkdir(parents=True, exist_ok=True)

        download_file(iot23_url, raw_iot23_zip)
    else:
        print(f"File {raw_iot23_zip} found!")

    labeled_files = get_files(iot23_dir)  # list of all labeled files
    if len(labeled_files) > 0:
        print('\nTotal number of labeled files: ', len(labeled_files))
    else:
        print('No unzipped labeled files found!')
        extract_file(raw_iot23_zip, iot23_dir)

    # consider only two small files for debugging purpose
    if config['debug']:
        debug_files = []
        for i in range(0, len(labeled_files)):
            if (show_file_size(labeled_files[i]) < 5.0):
                debug_files.append(labeled_files[i])
        assert len(labeled_files) > 0, \
            'No small files for debugging with size 5MB !'
        print('\nList of small files for debugging:', debug_files)

    cols = get_columns(labeled_files[0])
    cols.remove('fields')
    print('\nNumber of columns: ', len(cols))
    print('Columns: ', cols)
    assert len(cols) > 0, 'No columns found!'
    chunk_size = config['preprocess']['chunk_size']
    file2rows(config, labeled_files, cols, chunk_size)
