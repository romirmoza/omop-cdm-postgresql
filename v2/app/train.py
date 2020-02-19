import os
import psutil
import datetime
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn as skl
from joblib import dump
import xgboost as xgb
from automl import *

ROOT="/"

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.modelfile = ROOT+'model/xgb_model.joblib'
        self.train_features = 0
        self.process = psutil.Process(os.getpid())
        self.train = 0

        
    def load_data(self, filename):
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train load data start::Mem Usage {:.2f} MB".format(mem), flush = True)
        train = pd.read_csv(filename, compression='gzip', low_memory = False)
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
        self.train = train
        
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train load data end::Mem Usage {:.2f} MB".format(mem), flush = True)
        return


    def xgb_fit(self):
        '''
        apply XGB
        '''
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train fit start::Mem Usage {:.2f} MB".format(mem), flush = True)

        imbalance = int(round(self.train.shape[0] / self.train.death_in_next_window.sum()))
        random_state = 1234
        num_round = 500
        early_stop = round(num_round / 5)  # 20% of the full rounds

        params = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': ['auc'],
            'tree_method' : 'auto',
            'random_state' : random_state,
            'reg_lambda' : 1.0,
            'min_child_weight' : 1.0,
            'max_bin' : 256,
            'min_split_loss' : 0.01,
            'max_depth' : 15,
            'reg_alpha' : 0.0,
            'colsample_bylevel' : 1.0,
            'scale_pos_weight' : imbalance,
            'max_delta_step' : 0.0,
            'learning_rate' : 0.05,
            'n_estimators' : 1000,
            'num_parallel_tree' : 1,
            'colsample_bytree' : 0.7,
            'subsample' : 1.0,
            'missing': 0,
        }

        drop_features = ['death_in_next_window', 'window_id', 'person_id']

        xgb_model = None # clear out the xgb_model
        for w in self.train.window_id.unique():
            X = self.train[self.train.window_id == w] 
            y = self.train[self.train.window_id == w].death_in_next_window
            X = X.drop(drop_features, axis=1)
            self.train_features = X.columns.values
            X = np.array(X)
            y = np.array(y).ravel()
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            for i, (train_index, valid_index) in enumerate(cv.split(X, y)):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]
                evals_result ={}
                # Convert our data into XGBoost format
                d_train = xgb.DMatrix(X_train, y_train, feature_names=self.train_features)
                d_valid = xgb.DMatrix(X_valid, y_valid,  feature_names=self.train_features)
                watchlist = [(d_train, 'train'), (d_valid, 'valid')]

                xgb_model = xgb.train(params=params, dtrain=d_train, num_boost_round=num_round, 
                              evals=watchlist, evals_result=evals_result, 
                              early_stopping_rounds=early_stop, verbose_eval=False,
                              xgb_model=xgb_model)

                print("Best Score:%f, best iteration:%d, best ntree:%d" % 
                      (xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))
                gc.collect()
        

        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+\
              "Train AUC = "+str(evals_result['train']['auc'][xgb_model.best_ntree_limit])+\
              " Valid AUC = "+str(xgb_model.best_score)+\
              ' at '+str(xgb_model.best_ntree_limit), flush = True)

        dump(xgb_model, self.modelfile)

        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Train fit end::Mem Usage {:.2f} MB".format(mem), flush = True)
        return


if __name__ == '__main__':
    FOLDER = 'scratch/'
    FILE_STR = 'train_all.csv.gz'
    op = OmopParser()
    op.load_data(ROOT + FOLDER + FILE_STR)
    op.xgb_fit()
