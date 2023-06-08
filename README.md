# Bankruptcy Prediction

## Data
The dataset consists of:
* independent variables: financial indicators (~95) for selected Taiwan Companies (~6800)
* label: information whether company went bankrupt or not.

## Source
Data comes from the Taiwan Economic Journal including period of 1999–2009.

## Kaggle:
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

## Task
* To build a preprocessing pipeline, that will overcome the challenges with the data
* Find an appropriate classifier (ML models with different setups of hyperparameters) for bankruptcy prediction

## Challenges
* Low data quality (irrelevant data in some columns, highly correlated columns)
* Unbalanced data (classes distribution is ~1/30)

## Progress
1. Data preprocessing
2. Oversampling

![img](./train_test_split.png)
3. ML

3.1. Random Forrest

3.1.1 Test hyperparameters setups:
[link to visualizations](models/random-forrest-hyperparameters-dependencies.html)

...
