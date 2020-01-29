import os
import datetime
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn as skl
from joblib import dump
import xgboost as xgb

ROOT="/"

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+'Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    end_mem = df.memory_usage().sum() / 1024**2
    print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+'Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+'Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.modelfile = ROOT+'model/xgb_model.joblib'
        self.train_features = 0
        self.X = 0
        self.y = 0

    def load_data(self, filename):
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train load data start", flush = True)
        train = pd.read_csv(filename,low_memory = False)
        train = train.fillna(0)
        train.old = train.old.astype(int)
        for c in train.columns[train.columns.str.startswith('days')]:
            train[c] = train[c].astype(int)
        train = reduce_mem_usage(train)

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(train.race_concept_name)
        train.race_concept_name = label_encoder.transform(train.race_concept_name)
        train = train.fillna(0)
        X = train.drop(['death_in_next_window','person_id'], axis = 1)
        self.train_features = X.columns.values
        y = train[['death_in_next_window']]
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train load data end", flush = True)
        return


    def xgb_fit(self):
        '''
        apply XGB
        '''

        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train fit start", flush = True)

        params = {
            'eval_metric': 'auc',
            'tree_method' : 'auto',
            'random_state' : 1234,
            'reg_lambda' : 1.0,
            'min_child_weight' : 1.0,
            'max_bin' : 256,
            'min_split_loss' : 0.01,
            'max_depth' : 10,
            'reg_alpha' : 0.0,
            'colsample_bylevel' : 1.0,
            'scale_pos_weight' : 1.0,
            'max_delta_step' : 0.0,
            'learning_rate' : 0.05,
            'n_estimators' : 1000,
            'num_parallel_tree' : 1,
            'colsample_bytree' : 0.5,
            'subsample' : 1.0,
            'n_jobs': -1.0,
        }

        num_round = 2000

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)

        for train_index, valid_index in sss.split(self.X, self.y):
            X_train, X_valid = self.X[train_index], self.X[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]


        evals_result ={}
        d_train = xgb.DMatrix(X_train, y_train, feature_names=self.train_features)
        d_valid = xgb.DMatrix(X_valid, y_valid,  feature_names=self.train_features)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        xgb_model = xgb.train(params=params, dtrain=d_train, num_boost_round=num_round,
                      evals=watchlist, evals_result=evals_result,
                      early_stopping_rounds=200, verbose_eval=False)


        dump(xgb_model, self.modelfile)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+\
              "Train AUC = "+str(evals_result['train']['auc'][xgb_model.best_iteration])+\
              " Valid AUC = "+str(evals_result['valid']['auc'][xgb_model.best_iteration])+\
              ' at '+str(xgb_model.best_iteration), flush = True)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train fit end", flush = True)
        return


if __name__ == '__main__':
    FOLDER = 'scratch/'
    FILE_STR = 'train_all.csv'
    op = OmopParser()
    op.load_data(ROOT + FOLDER + FILE_STR)
    op.xgb_fit()
