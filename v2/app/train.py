import datetime
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn as skl
from joblib import dump
import xgboost as xgb
        
ROOT="/"

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.modelfile = ROOT+'model/xgb_model.joblib'
        self.train_features = 0
        self.d_train = 0

    def load_data(self, filename):
        print("Train load data start", flush = True)
        train = pd.read_csv(filename,low_memory = False)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(train.race_concept_name)
        train.race_concept_name = label_encoder.transform(train.race_concept_name)
        train = train.fillna(0)
        X = train.drop(['death_in_next_window','person_id'], axis = 1)
        self.train_features = X.columns.values
        y = train[['death_in_next_window']]
        X = np.array(X)
        y = np.array(y).ravel()
        self.d_train = xgb.DMatrix(X, y, feature_names=self.train_features)
        print("Train load data end", flush = True)
        return
        
        
    def xgb_fit(self):
        '''
        apply XGB
        '''

        print("Train fit start", flush = True)
        
        params = {
            'eval_metric': ['auc', 'error'],
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

        num_round = 1000

        evals_result ={}

        watchlist = [(self.d_train, 'train_full')]

        xgb_model = xgb.train(params, self.d_train, num_round, watchlist,
                      early_stopping_rounds=50, maximize=True, 
                      verbose_eval=False)


        dump(xgb_model, self.modelfile)
        print("Train fit end", flush = True)
        return
    
    
if __name__ == '__main__':
    FOLDER = 'scratch/'
    FILE_STR = 'train_all_nw.csv'
    op = OmopParser()
    op.load_data(ROOT + FOLDER + FILE_STR)
    op.xgb_fit()
