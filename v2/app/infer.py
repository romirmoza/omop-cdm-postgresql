import os
import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
import xgboost as xgb

ROOT = "/"

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
        self.person_id = 0
        self.d_test = 0

    def load_data(self, test_filename, train_filename):
        print(str(pd.datetime.now())+"::"+os.path.realpath(__file__)+"::"+"Infer load data start", flush = True)
        test = pd.read_csv(test_filename,low_memory = False)
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

        Y_pred = xgb_model.predict(self.d_test, ntree_limit=xgb_model.best_ntree_limit)
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
