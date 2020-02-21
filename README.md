# EHR DREAM Challenge - Patient Mortality Prediction
=================
- by tema EHR_noobs
- Romir Moza, Paul Perry,
- Navya Network, Inc

## Summary:

Gradient Boosting application for Mortality prediction.

## Background/Introduction

As EHR_noobs, our motivation was to learn the workflow of a competition on [DREAM Challenge]() and to familiarize ourselves with structured EHR data. Having joined the competition only towards the end of the final stage, our goal was to submit a model that could beat the baseline.

In this competition workflow, submitted docker models will be run on premises at the medical sites hosting the challenge EHR data. Analysts are not allowed access to the real data, only synthetic data, and no data can be returned by the model as it runs with the real data. Given this workflow and our time constraints, our thought was that an [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) approach, where the entire machine learning process is automated, would be the best way to build a model at the hosting site.

The problem is that very few commercial AutoML systems are deployable to a site; most are hosted services. Our approach, therefore, was to submit the synthetic data to an AutoML service, learn the best feature selection and model, and develop and package this model for deployment to the site.


## Methods

- Data ingestion
We started out using PostgreSQL for data management but quickly realized it wasn't all that necessary. We settled on ingesting data from the csv files provided, directly into our python model. We load in the patient data provided to us in the person, visit_occurence, condition_occurence, procedure_occurence, observation, and drug_exposure files and merge in based on person ids.
- Feature engineering
We aggregate the data for each person (visit, condition, procedure, drugs etc) and split the data for each person in time windows of 6 months. Our rationale was that data that is too far into the past is not meaningful to predicting mortality of the patient. We arrived at the window size of 6 months through extensive testing. Since most of these columns are categorical, we one hot encode them after aggregation. It's important to note though, that this method exasperates the problem of imbalanced classes (death, not death) in the data. For our target variable, "death_in_next_window" we lookup death date for persons who passed away within 6 months of the last time window.
- Feature selection
Running our first models, we realized that the feature space we had was too big for a model to run in a reasonable amount of time. So, we did regression analysis using Extra Trees Classifier, Random Forest Classifier, and Alternating conditional expectations (ACE) algorithms to generate feature importances and only selecting the top 200.
- Model selection
We used the autoML approach to find the models that performed the best on our training data. Tree based models seemed to do better than all the others and we eventually settled on XGBoost which performed the best.
- Cross-validation
Since the target classes are highly imbalanced we use a Stratified shuffles split to split the training data.
- Hyperparameter optimization
We then perform a randomized grid search on the hyperparameter space for our models.

## Conclusion/Discussion

The challenge was an interesting experience. First off, it gave us insight into the platform itself. This being our first challenge on synapse it took us some effort packaging the code for docker and running it on your servers.
We understand now that domain knowledge and the OMOP hierarchy needs to be used for feature engineering and selection to be able to get anywhere with the prediction model.
## References

Justin Guinney, [EHR DREAM Challenge - Patient Mortality Prediction](https://www.synapse.org/#!Synapse:syn18405991/wiki/589657),
[DOI: 10.7303/syn18405991](https://doi.org/10.7303/syn18405991), Collection published 2019 via Synapse.

## Authors Statement

- Romir Moza: feature engineering, feature extraction, feature selection, dimensionality reduction, model selection, MLOps.
- Paul Perry: methodology, tool selection, data exploration, model selection.
