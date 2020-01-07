#!/bin/bash
dropdb --if-exists commondatamodel
createdb commondatamodel
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ ddl.txt
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ load\ data.txt
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ pk\ indexes.txt
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ constraints.txt
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ denormalize.txt
psql commondatamodel -a -f OMOP\ CDM\ postgresql\ person_visit_occurence.txt
