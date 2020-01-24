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
COPY ./generate_training_data_nw.py .
COPY ./generate_test_data_nw.py .
COPY ./features.txt .

# Add executable permission to Bash scripts
RUN chmod +x train.sh infer.sh
