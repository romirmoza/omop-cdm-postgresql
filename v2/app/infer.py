import os
import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
import xgboost as xgb

ROOT = "/"

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.modelfile = ROOT+'model/xgb_model.joblib'
        self.person_id = 0
        self.d_test = 0

    def load_data(self, test_filename, train_filename):
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer load data start", flush = True)
        test = pd.read_csv(test_filename,low_memory = False)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(test.race_concept_name)
        test.race_concept_name = label_encoder.transform(test.race_concept_name)
        test = test.fillna(0)
        self.person_id = test.person_id
        # order the columnns of the test set like the train set
        train_features = pd.read_csv(train_filename, nrows=1).columns.values
        for feature in train_features:
            if feature not in test.columns:
                test[feature] = np.nan
        X = test[train_features]
        X = X.drop(['death_in_next_window','person_id'], axis = 1)
        feature_names=X.columns.values
        y = test[['death_in_next_window']]
        X = np.array(X)
        y = np.array(y).ravel()
        self.d_test = xgb.DMatrix(X, y, feature_names=feature_names)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer load data done", flush = True)
        return

    def xgb_predict(self):
        '''infer with XGBoost'''
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer start", flush = True)
        xgb_model =  load(self.modelfile)

        Y_pred = xgb_model.predict(self.d_test)
        output = pd.DataFrame(Y_pred,columns = ['score'])
        output_prob = pd.concat([self.person_id,output],axis = 1)
        output_prob.columns = ["person_id", "score"]
        output_prob.score = output_prob.score.clip(0.0,1.0)  # just in case
        output_prob.to_csv(ROOT+'output/predictions.csv', index = False)
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer finished", flush = True)

        return

if __name__ == '__main__':
    FOLDER ='scratch/'
    TEST_FILE_STR = 'test_all_nw.csv'
    TRAIN_FILE_STR = 'train_all_nw.csv'
    op = OmopParser()
    op.load_data(ROOT+FOLDER+TEST_FILE_STR, ROOT+FOLDER+TRAIN_FILE_STR)
    op.xgb_predict()
