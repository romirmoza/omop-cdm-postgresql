## generate test data

import pandas as pd
from datetime import timedelta
import re
import pickle

import warnings
warnings.filterwarnings('ignore')

rows_limit = 10000
concept_dir = '/app/concept_codes_final/'
training_dir = '/infer/'
features_filepath = 'features.txt'

def check_death_flag(x, window_size):
    if x.death_date - x.visit_start_date < window_size and x.death_date - x.visit_start_date >= timedelta(days = 0):
        return 1
    return 0

def visit_types_count(x):
    return pd.Series(dict(
        inpatient_visit_count  = (x.visit_concept_name == 'Inpatient Visit').sum(),
        outpatient_visit_count = (x.visit_concept_name == 'Outpatient Visit').sum(),
        er_visit_count         = (x.visit_concept_name == 'Emergency Room Visit').sum()
        ))

def agg_condition_concept_id(x, important_features_set):
    return pd.Series(dict(
        condition_concept_id_list  = ', '.join(set(x.condition_concept_id).intersection(important_features_set)),
        condition_type_concept_id_list  = ', '.join(set(x.condition_type_concept_id))
        ))

def agg_procedure_concept_id(x, important_features_set):
    return pd.Series(dict(
        procedure_concept_id_list  = ', '.join(set(x.procedure_concept_id).intersection(important_features_set)),
        procedure_type_concept_id_list  = ', '.join(set(x.procedure_type_concept_id))
        ))

def agg_drug_concept_id(x, important_features_set):
    return pd.Series(dict(
        drug_concept_id_list  = ', '.join(set(x.drug_concept_id).intersection(important_features_set)),
        drug_type_concept_id_list  = ', '.join(set(x.drug_type_concept_id))
        ))

def agg_observation_concept_id(x, important_features_set):
    return pd.Series(dict(
        observation_concept_id_list  = ', '.join(set(x.observation_concept_id).intersection(important_features_set)),
        observation_type_concept_id_list  = ', '.join(set(x.observation_type_concept_id))
        ))

def aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func, calc_death=0):
    if(calc_death):
        df['death_in_next_window'] = df.apply(lambda x: check_death_flag(x, window_size), axis=1)
        df['old'] = df.visit_start_date.dt.year - df.year_of_birth
    df[date_var] = df['death_date'] - df[date_var]
    df[date_var] = df[date_var].clip(upper=timedelta(days = 1000))
    df_agg = df.groupby(group_by_var).agg(agg_dict).rename(columns=rename_dict)
    apply_cols = df.groupby(group_by_var).apply(lambda x: apply_func(x))
    df_agg = df_agg.join(apply_cols)
    df_agg = df_agg.reset_index(drop=True)
    return df_agg

def generate_training_data_nw():
    print("Generate test data start", flush = True)
    filepath = training_dir + 'person.csv'
    print("Reading person data", flush = True)
    df_person = pd.read_csv(filepath, usecols = ['year_of_birth',
                                                 'ethnicity_concept_id',
                                                 'person_id',
                                                 'month_of_birth',
                                                 'day_of_birth',
                                                 'race_concept_id',
                                                 'gender_concept_id'], nrows=rows_limit)

    print("Reading visit_occurrence data", flush = True)
    filepath = filepath = training_dir + 'visit_occurrence.csv'
    df_visits = pd.read_csv(filepath, usecols=['person_id',
                                               'visit_start_date',
                                               'preceding_visit_occurrence_id',
                                               'visit_occurrence_id',
                                               'visit_end_date',
                                               'visit_concept_id',
                                               'visit_type_concept_id',
                                               'discharge_to_concept_id'], nrows=rows_limit)

    df_person_visits = pd.merge(df_person, df_visits, on=['person_id'], how='left')

    del df_person
    del df_visits

    filepath = concept_dir + 'all_concepts.csv'

    print("Reading all_concepts data", flush = True)
    df_concepts = pd.read_csv(filepath, usecols=['concept_name',
                                                 'concept_id',
                                                 'vocabulary_id'], nrows=rows_limit)

    df_concepts_race = df_concepts[df_concepts.vocabulary_id=='Race']
    df_concepts_race = df_concepts_race.drop(columns=['vocabulary_id'])
    df_concepts_race = df_concepts_race.rename(columns={'concept_id': 'race_concept_id',
                                                        'concept_name': 'race_concept_name'})

    df_person_visits_race = pd.merge(df_person_visits, df_concepts_race, on=['race_concept_id'], how='left')
    del df_person_visits

    df_concepts_visit = df_concepts[df_concepts.vocabulary_id=='Visit']
    df_concepts_visit = df_concepts_visit.drop(columns=['vocabulary_id'])
    df_concepts_visit = df_concepts_visit.rename(columns={'concept_id': 'visit_concept_id',
                                                          'concept_name': 'visit_concept_name'})

    df_person_visits_race_concepts = \
    pd.merge(df_person_visits_race, df_concepts_visit, on=['visit_concept_id'], how='left')

    filepath = training_dir + 'death.csv'
    print("Reading death data", flush = True)
    df_death = pd.read_csv(filepath, usecols=['person_id',
                                              'death_date',
                                              'death_datetime',
                                              'death_type_concept_id'], nrows=rows_limit)

    df = pd.merge(df_person_visits_race_concepts, df_death, on=['person_id'], how='left')

    df[['visit_start_date','visit_end_date', 'death_date']] = \
    df[['visit_start_date','visit_end_date', 'death_date']].apply(pd.to_datetime, format='%Y-%m-%d')
    df['visit_duration'] = df['visit_end_date'] - df['visit_start_date']

    df['visit_end_date'] = df['visit_end_date'].fillna(df['visit_start_date'])
    df['death_date'] = df['death_date'].fillna(pd.Timestamp.max)
    death_data = df[['person_id', 'death_date']]

    window_size = timedelta(days = 180)
    agg_dict = {'person_id': 'max',
                'year_of_birth': 'max',
                'visit_start_date': 'min',
                'ethnicity_concept_id': 'max',
                'race_concept_id': 'max',
                'gender_concept_id': 'max',
                'race_concept_name': 'max',
                'visit_occurrence_id': 'nunique',
                'visit_concept_name': 'count',
                'visit_duration': 'sum',
                'death_in_next_window': 'max',
                'old': 'max'}

    rename_dict = {'visit_occurrence_id': 'number_of_visits',
                   'visit_start_date': 'days_since_latest_visit'}

    group_by_var = 'person_id'
    date_var = 'visit_start_date'
    apply_func = visit_types_count

    training_data = \
    aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func, 1)
    training_data = training_data.drop(['year_of_birth'], axis=1)

    # Import
    f = open(features_filepath, "r")
    features = ''
    for x in f:
        features += x

    important_conditions = re.findall(r"condition_concept_([0-9]+)", features)
    important_procedures = re.findall(r"procedure_concept_([0-9]+)", features)
    important_drugs = re.findall(r"drug_concept_([0-9]+)", features)
    important_observations = re.findall(r"observation_concept_([0-9]+)", features)

    # Merge with condition_occurrence
    print("Reading condition_occurrence data", flush = True)
    filepath = training_dir + 'condition_occurrence.csv'
    df = pd.read_csv(filepath, usecols = ['condition_occurrence_id',
                                          'person_id',
                                          'condition_concept_id',
                                          'condition_start_date',
                                          'condition_end_date',
                                          'condition_type_concept_id',
                                          'condition_status_concept_id',
                                          'visit_occurrence_id'], nrows=rows_limit)

    df['condition_end_date'] = df['condition_end_date'] if not 'NaT' else df['condition_start_date']
    df['condition_concept_id'] = df['condition_concept_id'].astype('Int64')
    df['condition_type_concept_id'] = df['condition_type_concept_id'].astype('Int64')
    df['condition_status_concept_id'] = df['condition_status_concept_id'].astype('Int64')
    df['condition_concept_id'] = df['condition_concept_id'].apply(str)
    df['condition_type_concept_id'] = df['condition_type_concept_id'].apply(str)
    df['condition_status_concept_id'] = df['condition_status_concept_id'].apply(str)
    df[['condition_start_date','condition_end_date']] = \
    df[['condition_start_date','condition_end_date']].apply(pd.to_datetime, format='%Y-%m-%d')

    df = pd.merge(df, death_data, on=['person_id'], how='left')

    agg_dict = {'person_id': 'max',
                'condition_start_date': 'min',
                'condition_status_concept_id': 'max'}

    rename_dict = {'condition_start_date': 'days_since_latest_condition'}

    group_by_var = 'person_id'
    date_var = 'condition_start_date'
    important_features_set = set(important_conditions)
    apply_func = lambda x: agg_condition_concept_id(x, important_features_set)

    df.condition_start_date = pd.to_datetime(df.condition_start_date, format='%Y-%m-%d')
    cond_occur_data = \
    aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func)

    training_data = pd.merge(training_data, cond_occur_data, on=['person_id'], how='left')
    del cond_occur_data

    # Merge with procedure_occurrence
    filepath = training_dir + 'procedure_occurrence.csv'
    print("Reading procedure_occurrence data", flush = True)
    df = pd.read_csv(filepath, usecols = ['procedure_occurrence_id',
                                          'person_id',
                                          'procedure_concept_id',
                                          'procedure_date',
                                          'procedure_type_concept_id',
                                          'visit_occurrence_id'], nrows=rows_limit)

    df['procedure_concept_id'] = df['procedure_concept_id'].astype('Int64')
    df['procedure_type_concept_id'] = df['procedure_type_concept_id'].astype('Int64')
    df['procedure_concept_id'] = df['procedure_concept_id'].apply(str)
    df['procedure_type_concept_id'] = df['procedure_type_concept_id'].apply(str)

    df = pd.merge(df, death_data, on=['person_id'], how='left')

    agg_dict = {'person_id': 'max',
                'procedure_date': 'min'}

    rename_dict = {'procedure_date': 'days_since_latest_procedure'}

    group_by_var = 'person_id'
    date_var = 'procedure_date'
    important_features_set = set(important_procedures)
    apply_func = lambda x: agg_procedure_concept_id(x, important_features_set)

    df.procedure_date = pd.to_datetime(df.procedure_date, format='%Y-%m-%d')
    procedure_occur_data = \
    aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func)

    training_data = pd.merge(training_data, procedure_occur_data, on=['person_id'], how='left')
    del procedure_occur_data

    # Merge with drug_exposure
    filepath = training_dir + 'drug_exposure.csv'
    print("Reading drug_exposure data", flush = True)
    df = pd.read_csv(filepath, usecols = ['drug_exposure_id',
                                          'person_id',
                                          'drug_concept_id',
                                          'drug_exposure_start_date',
                                          'drug_type_concept_id',
                                          'quantity',
                                          'visit_occurrence_id'], nrows=rows_limit)

    df['drug_concept_id'] = df['drug_concept_id'].astype('Int64')
    df['drug_type_concept_id'] = df['drug_type_concept_id'].astype('Int64')
    df['drug_concept_id'] = df['drug_concept_id'].apply(str)
    df['drug_type_concept_id'] = df['drug_type_concept_id'].apply(str)

    df = pd.merge(df, death_data, on=['person_id'], how='left')

    agg_dict = {'person_id': 'max',
                'drug_exposure_start_date': 'min',
                'quantity': 'sum'}

    rename_dict = {'drug_exposure_start_date': 'days_since_latest_drug_exposure',
                   'quantity': 'total_quantity_of_drugs'}

    group_by_var = 'person_id'
    date_var = 'drug_exposure_start_date'
    important_features_set = set(important_drugs)
    apply_func = lambda x: agg_drug_concept_id(x, important_features_set)

    df.drug_exposure_start_date = pd.to_datetime(df.drug_exposure_start_date, format='%Y-%m-%d')
    drug_exposure_data = \
    aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func)

    training_data = pd.merge(training_data, drug_exposure_data, on=['person_id'], how='left')
    del drug_exposure_data

    # Merge with oberservations
    filepath = training_dir + 'observation.csv'
    print("Reading observation data", flush = True)
    df = pd.read_csv(filepath, usecols = ['observation_id',
                                          'person_id',
                                          'observation_concept_id',
                                          'observation_date',
                                          'observation_type_concept_id',
                                          'value_as_string',
                                          'value_as_concept_id'], nrows=rows_limit)

    df['observation_concept_id'] = df['observation_concept_id'].astype('Int64')
    df['observation_type_concept_id'] = df['observation_type_concept_id'].astype('Int64')
    df['observation_concept_id'] = df['observation_concept_id'].apply(str)
    df['observation_type_concept_id'] = df['observation_type_concept_id'].apply(str)

    df = pd.merge(df, death_data, on=['person_id'], how='left')

    agg_dict = {'person_id': 'max',
                'observation_date': 'min'}

    rename_dict = {'observation_date': 'days_since_latest_observation'}

    group_by_var = 'person_id'
    date_var = 'observation_date'
    important_features_set = set(important_observations)
    apply_func = lambda x: agg_observation_concept_id(x, important_features_set)

    df.observation_date = pd.to_datetime(df.observation_date, format='%Y-%m-%d')
    observation_data = \
    aggregate_data(df, window_size, group_by_var, date_var, agg_dict, rename_dict, apply_func)

    training_data = pd.merge(training_data, observation_data, on=['person_id'], how='left')
    del observation_data

    # make a copy, preserve the original
    train = training_data.copy()
    col_num = train.shape[1]

    # unroll the _list columns and one-hot encode them
    print("Unrolling cols", flush = True)
    lists = [c for c in train.columns if '_list' in c]
    for idx, row in train.iterrows():
        for l in lists:
            l_str = '_'.join(l.split('_')[:2])+'_'
            l_items = row[l]
            if isinstance(l_items, str):
                l_items = l_items.split(',')
                if isinstance(l_items, list) and l_items != ['']:
                    for c in l_items:
                            train.loc[idx,l_str+str(c).strip()] = 1

    train[col_num:].fillna(0, inplace=True)
    train = train.drop(lists, axis=1)
    date_cols = [c for c in train.columns if 'days' in c]
    for c in date_cols:
        train[c] = pd.to_timedelta(train[c]).dt.days
    train.visit_duration = pd.to_timedelta(train.visit_duration).dt.days
    train.race_concept_name = train.race_concept_name.replace(to_replace=0, value='Unknown')
    train.race_concept_name = train.race_concept_name.fillna('Unknown')

    train.to_csv('/scratch/test_all_nw.csv', index=False)
    print("Generate test data end", flush = True)
    return 0

if __name__ == '__main__':
    generate_training_data_nw()
