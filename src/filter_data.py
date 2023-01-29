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
@File - filtering of the data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import (cross_val_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.utility import load_config, load_data


def one_hot_encoding(y):
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform([y])
    return y_onehot


def apply_model(X, y_onehot):
    print('\nModel training performance metrics')
    clf = DecisionTreeClassifier(max_depth=None, 
                                 min_samples_split=2,
                                 random_state=41)
    scores = cross_val_score(clf, X, y_onehot, cv=5)
    print('Decision tree performance mean: ', scores.mean())
    print('Decision tree performance scores:\n', scores)

    clf = RandomForestClassifier(n_estimators=10, 
                                 max_depth=None,
                                 min_samples_split=2, 
                                 random_state=41)
    scores = cross_val_score(clf, X, y_onehot, cv=5)
    print('Random forest performance mean: ', scores.mean())
    print('Random forest performance scores:\n', scores)

    clf = ExtraTreesClassifier(n_estimators=10, 
                               max_depth=None,
                               min_samples_split=2,
                               random_state=41)
    scores = cross_val_score(clf, X, y_onehot, cv=5)
    print('ExtraTrees performance mean: ', scores.mean())
    print('ExtraTrees performance scores:\n', scores)
    print('\n')

    # clf = AdaBoostClassifier(n_estimators=100)
    # scores = cross_val_score(clf, X, y_onehot, cv=5)
    # print(scores.mean())

    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    #                                  max_depth=1, random_state=0).fit(X, y_onehot)
    # clf.score(X, y_onehot)


def plot_3D_PCA(df, FIG_DIR):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(
        df.drop(['multi_label'], axis=1))
    principalDf = pd.DataFrame(
        data=principalComponents, columns=['pc1', 'pc2', 'pc3'])
    finalDf = pd.concat([principalDf, df[['multi_label']]], axis=1)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('pc1', fontsize=15)
    ax.set_xlabel('pc2', fontsize=15)
    ax.set_xlabel('pc3', fontsize=15)
    ax.set_title('PCA with 3 components', fontsize=20)
    targets = list(set(df.multi_label))
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['multi_label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep,
                   'pc2'], finalDf.loc[indicesToKeep, 'pc3'], c=color)
    ax.legend(targets, loc=2)
    ax.grid()
    fig.tight_layout()
    fig.savefig(FIG_DIR + '3D_PCA.pdf', format='pdf')


def plot_corr(df, FIG_DIR):
    # Using Pearson Correlation
    plt.figure(figsize=(20, 18))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds,
                fmt='.2f', annot_kws={'size': 8})
    plt.title('Heatmwap showing correlation on IoT-23 dataset features', fontsize=14)
    plt.savefig(FIG_DIR + 'iot-23-heatmap.pdf', format='pdf')
    # plt.show()


def get_selected_features(df):
    """
    Filtering steps to remove duplicate rows:
    """
    # Step 1. drop duplicate rows for selected features and prevents not removing all duplicate rows by keep = 'first'.
    total_rows = int(df.shape[0])
    df = df.drop_duplicates(
        keep='first').reset_index(drop=True)
    num_rows_selected = int(df.shape[0])
    print(f'#rows after dropping duplicates and keeping first of the duplicates: \
        {num_rows_selected} rows (filtered {total_rows - num_rows_selected} rows)')

    # Step 2. keep = False drops all duplicates not keeping the default first one even
    # because we should ignore all the ambiguous rows.
    df = df.drop_duplicates(subset=fet_list,keep=False).reset_index(drop=True)
    print(f'#rows after dropping ambiguous rows (same features in different classes): \
        {df.shape[0]} rows (filtered {num_rows_selected - int(df.shape[0])} rows)')

    # Step 3. drop columns with same values in all rows
    # dup_cols = len(df) - len((df.T.iloc[1:]).drop_duplicates().T) # this line raises a memory issue and killed the job
    # print('#columns with same values in all rows: ', dup_cols)
    return df

# clean dataset, https://stackoverflow.com/a/46581125
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be of pd.DataFrame type"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]


def fit_model(df, model, plot_file):
    df = clean_dataset(df)
    x = df.drop(['multi_label'], axis=1)
    y = list(df['multi_label'])
    print('\n' + '-'*70)
    print('\t\tFitting the model...')
    print('-'*70)
    model.fit(x, y)
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 6))
    plt.savefig(plot_file, format='pdf')
    # apply_model(x, y)
    return model


if __name__ == '__main__':
    # Load config
    config = load_config('config.yaml')
    num_features = config['preprocess']['num_features']
    data_file = config['data_dir'] +  config['preprocess']['normalized_data']
    save_csv = config['data_dir'] + config['preprocess']['processed_data']
    FIG_DIR = config['result_dir'] + 'data_plots/'

    # create folders if not exist
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(save_csv.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)

    # loading the data for filtering
    df = load_data(data_file)
    # df = df.drop(['label'], axis=1)

    print('\n***************************** Complete Features **************************************')
    # df = df.drop_duplicates()  # drop duplicate rows
    # df = df.drop(filter_columns(df), axis=1)
    print('#Shape after dropping duplicates:', df.shape)
    print('\nSample data: \n', df.head(2))

    # plot_3D_PCA(df, FIG_DIR)

    model = ExtraTreesClassifier()
    
    # with all features
    file_plot = FIG_DIR + 'iot-23-features.pdf'
    model = fit_model(df, model, file_plot)
    
    # selecting important features
    cols = [x for x in list(df.columns) if x != 'multi_label']
    feat_importances = pd.Series(model.feature_importances_, index=cols)

    feat_importances.nlargest(len(cols)).plot(
        kind='barh',
        title='Feature importances (ExtraTrees) with 50 most important features')

    fet_list = (feat_importances.nlargest(num_features)).index
    df_select = df[fet_list].copy()
    df_select['multi_label'] = list(df['multi_label'])

    print('\n****************************** Selected Features **************************************\n')
    df_select = get_selected_features(df_select)

    feat_importances.nlargest(num_features).plot(
        kind='barh',
        title='Top {num_features} feature importances (ExtraTrees)')
    
    file_plot = FIG_DIR + 'iot-23-features-importance_full.pdf'
    model = fit_model(df_select, model, file_plot)
    # plot_corr(df_select, FIG_DIR)
    df_select.to_csv(save_csv, index=False)
    # df.to_csv('data/processed/all_features_with_dups.csv')

    print('\nThe feature-importances plot is saved at: ', file_plot)
    print('Shape of the filtered data: ', df_select.shape)
    print('The processed/filtered data is saved at: ', save_csv)
