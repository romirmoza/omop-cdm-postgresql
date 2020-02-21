#!/usr/bin/env bash
# python /app/generate_death_data.py 'train' '/train/' 'train_death_last.csv.gz'
# python /app/generate_variable_lookback_data.py 'train' '/train/' 'train_all.csv.gz'
python /app/generate_data.py 'train' '/train/' 'train_all.csv.gz'
python /app/train.py
