import os
import psutil
import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
import xgboost as xgb
from automl import *

ROOT = "/"
GPU=True

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.modelfile = ROOT+'model/xgb_model.joblib'
        self.process = psutil.Process(os.getpid())
        self.person_id = 0
        self.d_test = 0

    def load_data(self, test_filename, train_filename):
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer load data start::Mem Usage {:.2f} MB".format(mem),  flush = True)
        test = pd.read_csv(test_filename, compression='gzip', low_memory = False)
        test = test.fillna(0)
        test.old = test.old.astype(int)
        for c in test.columns[test.columns.str.startswith('days')]:
            test[c] = test[c].astype(int)
        test = reduce_mem_usage(test)

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(test.race_concept_name)
        test.race_concept_name = label_encoder.transform(test.race_concept_name)
        test = test.fillna(0)
        self.person_id = test.person_id
        # order the columnns of the test set like the train set
        train_features = pd.read_csv(train_filename, compression='gzip', nrows=1).columns.values
        for feature in train_features:
            if feature not in test.columns:
                test[feature] = np.nan
        X = test[train_features]
        X = X.drop(['death_in_next_window','window_id','person_id'], axis = 1)
        feature_names=X.columns.values
        y = test[['death_in_next_window']]
        X = np.array(X)
        y = np.array(y).ravel()
        self.d_test = xgb.DMatrix(X, y, feature_names=feature_names)
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer load data done::Mem Usage {:.2f} MB".format(mem),  flush = True)
        return

    def xgb_predict(self):
        '''infer with XGBoost'''
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer start::Mem Usage {:.2f} MB".format(mem),  flush = True)
        xgb_model =  load(self.modelfile)

        Y_pred = xgb_model.predict(self.d_test, ntree_limit=xgb_model.best_ntree_limit)
        output = pd.DataFrame(Y_pred,columns = ['score'])
        output_prob = pd.concat([self.person_id,output],axis = 1)
        output_prob.columns = ["person_id", "score"]
        output_prob.score = output_prob.score.clip(0.0,1.0)  # just in case
        output_prob.to_csv(ROOT+'output/predictions.csv', index = False)
        mem = self.process.memory_info()[0]/(1024**2)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer finished::Mem Usage {:.2f} MB".format(mem),  flush = True)

        return

if __name__ == '__main__':
    FOLDER ='scratch/'
    TEST_FILE_STR = 'test_all.csv.gz'
    TRAIN_FILE_STR = 'train_all.csv.gz'
    op = OmopParser()
    op.load_data(ROOT+FOLDER+TEST_FILE_STR, ROOT+FOLDER+TRAIN_FILE_STR)
    op.xgb_predict()
