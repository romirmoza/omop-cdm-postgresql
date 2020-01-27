docker run \
       -v /Users/paulperry/mortality_prediction_docker_model/v2/train:/train:ro \
       -v /Users/paulperry/mortality_prediction_docker_model/v2/data:/data:ro \
       -v /Users/paulperry/mortality_prediction_docker_model/v2/scratch:/scratch:rw \
       -v /Users/paulperry/mortality_prediction_docker_model/v2/model:/model:rw \
docker.synapse.org/syn21445804/ehr_xgb:v2 bash /app/train.sh