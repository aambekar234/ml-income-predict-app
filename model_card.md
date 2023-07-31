## Model Details
This model predicts whether income exceeds 50k based on census data.

## Intended Use
Predict if the salary would exceed 50K or not by providing certain data points. 

## Training Data
Information for the data used for training this model can be found at [here](https://archive.ics.uci.edu/dataset/20/census+income)

## Evaluation Data
20% of original data is used for Evaluation. One hot encoding is used for encoding categories and binary encoding is used to encode the labels. 

## Metrics


## Ethical Considerations
The dataset contains personal information such as race, grography, gender and age which may potentially descriminate against certain group of people.

## Caveats and Recommendations
The data used to train this model is not balanced across all features. For instance, there are fewer data points for certain countries and races, which can lead to bias towards underrepresented groups. To address this, we need to capture more data points to achieve balance among all groups.
