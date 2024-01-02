# Bankruptcy Prediction

## Data
The dataset consists of:
* independent variables: 95 financial indicators  for selected Taiwan Companies (~6800)
* label: information whether company went bankrupt or not.

## Source
Taiwanese Bankruptcy Prediction<br>
https://doi.org/10.24432/C5004D<br>
Taiwan Economic Journal 2020<br>

## Kaggle:
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

## Task
* Build an optimal classifier for bankruptcy prediction.

## Challenges
* Low data quality (irrelevant data in some columns, highly correlated columns)
* Unbalanced data (classes distribution is ~1/30)

## Pipeline
1. Data exploration
2. Data preprocessing (data cleaning, remove correlated columns, oversampling)

![img](./train_test_split.png)

3. ML

Find optimal hyperparameters for:<br>

3.1. Random Forrest <br>
3.2. XGBoost<br>
3.3. LightGBM<br>

4. Summary
