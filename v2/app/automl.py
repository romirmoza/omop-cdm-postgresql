###
#
# Util functions for automl
#
###

import os
import sys
import datetime
import gc
# import GPUtil


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load  # for saving models
import xgboost as xgb

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def print_manifest():
    print(pd.datetime.now())
    print('OS: '+os.uname().sysname)
    print('Python: '+sys.version)
    print('matplotlib: '+mpl.__version__)
    print('numpy: '+np.__version__)
    print('pandas: '+pd.__version__)
    print('sklearn: '+sklearn.__version__)
    print('xgboost: '+xgb.__version__)


def xgb_progressbar(rounds=1000):
    """Progressbar for xgboost using tqdm library.
    https://programtalk.com/python-examples/tqdm/
    example: model = xgb.train(params, X_train, 1000, callbacks=[xgb_progressbar(100), ])
    """
    from tqdm.auto import tqdm
    pbar = tqdm(total=rounds)

    def callback(_, ):
        pbar.update(1)

    return callback


def cols_toint(df):
    '''cast days_ columns to int'''
    df.old = df.old.astype(int)
    for c in df.columns[df.columns.str.startswith('days')]:
        df[c] = df[c].astype(int)
    return df


def reduce_mem_usage(df):
    '''https://www.mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas/'''
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type :
                c_min = df[col].min()
                c_max = df[col].max()
                c_unique = len(df[col].unique())

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)                    
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

                if c_min == 0.0 and c_max == 1.0 and c_unique == 2:
                    df[col] = df[col].astype(bool)
                    pass
                
#                 if 2 > c_unique and c_unique < 21: # convert to categorical - needs more work
#                     df[col] = pd.Categorical(df[col])


    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def find_dup_cols(df_in):
    '''
    Find duplicate columns:
    >>> df = pd.DataFrame([[1,1,4], [2,2,5], [3,3,6]], columns=['a','b','c']); find_dup_cols(df)
    ['b', 'c']
    '''
    from tqdm import tqdm

    df = df_in.copy(deep=True)
    i = 1
    dupes = []
    for c in tqdm(df.columns):
        if c not in df.columns: # a column could have been deleted
            pass
        for d in df.columns[i:]: # for every subsequent column
            if len(pd.Series(list(zip(df[c], df[d]))).unique()) == len(df[c].unique()) == len(df[d].unique()):
                dupes += d
                df = df.drop(d, axis=1)
        i += 1
    return dupes


#
#
#
def plot_perf_i(evals_result, nth=10): 
    '''interactive plot model performance over epochs interactive'''
    from IPython.display import display, clear_output
    fig, ax1 = plt.subplots(1,1, figsize=(10,5))
#    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))
    plt.ion()
    display(fig)
    counter = 0
    
    def callback(_,): #, fig, ax1, ax2):
        if evals_result == {}: # skip the empty dict  
            return callback

        epochs = len(evals_result['train']['auc'])
        if epochs % nth != 0:  # only plasdfot every nth iteration
            return callback

        clear_output(wait=True)  # plot over the previous picture
        x_axis = range(0, epochs)
        ax1.plot(x_axis, evals_result['train']['auc'], label='Train ', c='g')
        ax1.plot(x_axis, evals_result['valid']['auc'], label='Valid ', c='b')
        ax1.ticklabel_format(useOffset=False, style='plain')
        # ax1.legend()  # legends are messy and result in duplicates
        ax1.set_ylabel('AUC')
        ax1.set_title('XGBoost AUC ')
        ax1.grid(True)

        # needs to be generalized ??? 
#         ax2.plot(x_axis, evals_result['train']['error'], label='Train ', c='g')
#         ax2.plot(x_axis, evals_result['valid']['error'], label='Valid ', c='b')
#         ax2.ticklabel_format(useOffset=False, style='plain')
#         # ax2.legend()
#         ax2.set_ylabel('Classification Error')
#         ax2.set_title('XGBoost Classification Error')
#         ax2.grid(True)

        display(fig)
        plt.close()

    return callback

#
#
#
def plot_perf(evals_result, best):
    ''' plot all model performance metrics over epochs'''
    keys = list(evals_result.keys())
    metrics = list(evals_result[keys[0]].keys())
    epochs = len(evals_result[keys[0]][metrics[0]])
    num_plots = len(metrics)
    x_axis = list(range(0, epochs))
    fig, axs = plt.subplots(1, num_plots, figsize=(6*num_plots,3))
    if num_plots == 1: # in the case there is only one metric, then subplot doesn't return a list of ax's, so we create one
        axs = [axs]
    for p in range(num_plots):
        axs[p].plot(x_axis, evals_result[keys[0]][metrics[p]], 
                    label=keys[0]+' '+str(evals_result[keys[0]][metrics[p]][best]), c='g')
        axs[p].plot(x_axis, evals_result[keys[1]][metrics[p]], 
                    label=keys[1]+' '+str(evals_result[keys[1]][metrics[p]][best]), c='b')
        axs[p].ticklabel_format(useOffset=False, style='plain')
        axs[p].axvline(x=best, color='r', label='best '+str(best))
        axs[p].grid(True)
        axs[p].legend()
        axs[p].set_ylabel(metrics[p])
        axs[p].set_title(metrics[p])
    plt.show()



def buildROC(target_test,test_preds,label='', color='b'):
    '''Compute micro-average ROC curve and ROC area
        could be replaced by sklearn.metrics.plot_auc_curve in 0.22
    '''
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.rcParams["figure.figsize"] = (6,6)
    plt.plot(fpr, tpr, 'b', label = label+' AUC = %0.4f' % roc_auc, color=color)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('AUC')
    return 
