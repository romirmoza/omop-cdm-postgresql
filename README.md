EHR DREAM Challenge - Patient Mortality Prediction by EHR_noobs team
=================P
Romir Moza, Paul Perry, 
Navya Network, Inc

## Summary:
Gradient Boosting application for Mortality prediction.

## Background/Introduction
(suggested limit 200 words)

As EHR_noobs, our motivation was to learn the workflow of a
competition on [DREAM Challenge]() and to familiarize ourselves with
structured EHR data.  Having joined the competition only towards the
end of the final stage, our goal was to submit a model that could beat
the baseline.

In this competition workflow, submitted docker models will be run on
premises at the medical sites hosting the challenge EHR
data. Analysist are not allowed access to the real data, only
synthetic data, and no data can be returned by the model as it runs
with the real data. Given this workflow and our time constraints, our
thought was that an
[AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning)
approach, where the entire machine learning process is automated,
would be the best way to build a model at the hosting site.

The problem is that very few commercial AutoML systems are deployable
to a site; most are hosted services. Our approach therefore was to
submit the synthetic data to an AutoML service, learn the best feature
selection and model, and develop and package this model for deployment
to the site.

<!--
Please try to address the following points:

What is the motivation for your approach? This will include any previous work and observations that you have made about the data to suggest your approach is a good one. Provide the reader with an intuition of how you approached the problem
What is the underlying methodology used (e.g., SVM or regression)?
Where there any novel approaches taken in regards to
-->

## Methods
<!--
(suggested limit 800 words)

The methods should cover a full description of your methods so a reader can reproduce them. Please cover how you processed the data, if any data was imputed or manipulated in any way (e.g., you mapped data onto pathways or combined different datasets), the underlying algorithm, any modifications to the underlying method of importance, the incorporation of outside data, and the approach to predict submitted data.
If you submitted multiple predictions, please specify which is the difference among them (e.g. only parameters tuning or different algorithms). If needed, you can decide to write one sub-paragraph for each submission.
-->
- data imputation
- feature engineering
- feature selection
- cross validation
- model selection
- hyperparameter optimization
- final prediction

## Conclusion/Discussion

<!--
(suggested limit 200 words)

This section should include a short summary and any insights gained during the algorithm. For example, which dataset was most informative? You can include future directions. You may also add some discussion on the general performance of your methodology (if you wish) and if there were pitfalls, what are they?
-->

## References

<!--
(suggested limit 10 references)

Don't forget to reference your specific challenge (e.g. NIEHS-NCATS-UNC DREAM Toxicogenetics Challenge (syn1761567)).
-->
Justin Guinney, [EHR DREAM Challenge - Patient Mortality Prediction](https://www.synapse.org/#!Synapse:syn18405991/wiki/589657)
[DOI: 10.7303/syn18405991](https://doi.org/10.7303/syn18405991), Collection published 2019 via Synapse.

## Authors Statement
<!-- Please list all author's contributions -->
Romir Moza: feature engineering, feature extraction, feature selection, dimensionality reduction, model selection, MLOps.
Paul Perry: methodology, tool selection, data exploration, model selection.


