date
docker run \
-v /home/user/mortality/mortality_prediction_docker_model/v2/infer:/infer:ro \
-v /home/user/mortality/mortality_prediction_docker_model/v2/scratch:/scratch:rw \
-v /home/user/mortality/mortality_prediction_docker_model/v2/output:/output:rw \
-v /home/user/mortality/mortality_prediction_docker_model/v2/model:/model:rw \
docker.synapse.org/syn21445804/ehr_xgb:v2 bash /app/infer.sh
date
