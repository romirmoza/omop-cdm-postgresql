FROM python:3.5

# Install dependencies
RUN pip install pandas \
  numpy \
  sklearn \
  matplotlib \
  datetime \
  xgboost \
  keras \
  tensorflow \
  eli5 \
  joblib
RUN pip install psutil

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY ./train.py .
COPY ./infer.py .

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY ./train.sh .
COPY ./infer.sh .

# Copy Support files
COPY ./generate_data.py .
COPY ./automl.py .
COPY ./features.txt .
RUN mkdir concept_codes_final
COPY ./concept_codes_final/all_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/condition_occurrence_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/drug_exposure_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/measurement_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/person_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/procedure_occurrence_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/README.md ./concept_codes_final
COPY ./concept_codes_final/visit_occurrence_concepts.csv ./concept_codes_final
COPY ./concept_codes_final/observation_concepts.csv ./concept_codes_final

# Add executable permission to Bash scripts
WORKDIR /app
RUN chmod +x train.sh infer.sh
