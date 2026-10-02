# Wine Classification with XGBoost

Supervised multiclass classification of the **UCI Wine dataset** using **XGBoost**.

This project explores the complete workflow of a supervised machine learning classification problem, from exploratory data analysis to model evaluation and interpretation.

## Dataset

The project uses the **Wine Dataset** from the UCI Machine Learning Repository (ID 109).

The dataset contains:

- 178 observations
- 13 numerical features
- 3 wine classes
- No missing values

The features correspond to different physicochemical measurements of the wines.

Dataset source:  
https://archive.ics.uci.edu/dataset/109/wine

## Project Workflow

The notebook follows these main steps:

1. **Exploratory Data Analysis (EDA)**
   - Dataset structure and descriptive statistics
   - Missing and duplicated values
   - Class distribution
   - Feature distributions
   - Correlation analysis

2. **Data Preparation**
   - Separation of features and target
   - Target encoding for XGBoost
   - Stratified train/test split

3. **Model Training**
   - Multiclass classification using `XGBClassifier`
   - 80/20 stratified train/test split

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Classification report
   - Confusion matrix
   - Stratified 5-fold cross-validation

5. **Model Interpretation**
   - XGBoost feature importance analysis

## Model

The classification model is based on **XGBoost**:

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)
