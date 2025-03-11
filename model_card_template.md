# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier trained on Census Bureau data. It predicts whether an individual earns more than $50K per year based on features like age, education, occupation, and other demographic factors.

## Intended Use
The model is designed for classification tasks where the goal is to predict income levels based on structured demographic data. It is useful for applications such as social studies, economic analysis, and workforce research.

## Training Data
The dataset used for training is sourced from publicly available Census Bureau data. The dataset includes various demographic and work-related attributes, such as:

* Age

* Workclass

* Education

* Marital Status

* Occupation

* Relationship

* Race

* Sex

* Native Country

* Hours per week

* Capital Gain/Loss

The target variable (salary) is binary: <=50K or >50K.

## Evaluation Data
The dataset is split into 80% training and 20% testing. The same preprocessing pipeline is applied to both datasets. Evaluation is performed using the test set to ensure generalization of the model.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model was evaluated using precision, recall, and F1-score, which are critical metrics for classification problems where class imbalance may exist.

* Precision: 0.7419

* Recall: 0.6384

* F1-score: 0.6863
* 
## Ethical Considerations
* Bias in Data: The dataset is sourced from census data, which may contain inherent biases based on historical social and economic factors.

* Fairness: It is important to monitor the model’s performance across different demographic slices to ensure that no group is disproportionately impacted.

* Interpretability: The model is a Random Forest, which, while more interpretable than deep learning models, still lacks the transparency of simpler models like logistic regression.
* 
## Caveats and Recommendations
* Feature Importance: Users should analyze feature importance to understand what drives predictions.

* Generalization: The model is trained on Census data and may not generalize well to datasets with different distributions.

* Deployment Monitoring: If deployed in a real-world setting, the model’s performance should be continuously monitored for data drift and fairness.


