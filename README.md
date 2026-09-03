# Real Estate Price Prediction

Machine learning project focused on predicting real estate prices using linear regression and regularization techniques.

The project explores the use of **Linear Regression, Ridge Regression, Lasso Regression and Elastic Net** to evaluate how regularization affects model performance, complexity and generalization.

## Project Objective

The goal is to build and compare regression models for predicting property prices and investigate whether regularization techniques can improve model stability and reduce overfitting.

The analysis focuses on:

- Data preprocessing and standardization
- Correlation analysis between features
- Linear and polynomial regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Cross-validation
- Mean Squared Error (MSE)
- R² score
- Model coefficient analysis
- Learning curves
- Residual analysis

## Dataset

The project uses a housing dataset containing information about residential properties.

The target variable is:

- `price` — property price

The main features include:

- `area` — property area
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `stories` — number of floors
- `mainroad` — whether the property is connected to a main road
- `guestroom` — whether the property has a guest room
- `basement` — whether the property has a basement
- `hotwaterheating` — whether the property has hot water heating
- `airconditioning` — whether the property has air conditioning
- `parking` — number of parking spaces
- `prefarea` — whether the property is located in a preferred area
- `furnishingstatus` — furnishing condition of the property

The dataset is based on the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

## Data Preparation

The dataset is loaded using Pandas and checked for missing values.

The input features are separated from the target variable and then divided into training and test sets using a 70/30 split.

The numerical features are standardized using `StandardScaler` before training the regularized models.

A correlation matrix is also used to investigate relationships between the input variables.

## Models

### Linear Regression

A baseline linear regression model is evaluated first.

Polynomial features with degrees from 0 to 6 are tested to investigate whether increasing model complexity improves predictive performance.

The results are evaluated on both the training and test datasets using:

- Mean Squared Error (MSE)
- R² score

The experiments show that increasing the polynomial degree does not provide a meaningful improvement over the first-degree model.

### Ridge Regression

Ridge Regression is used to introduce L2 regularization.

Different values of the regularization parameter `alpha` are evaluated using `RidgeCV` in order to select a suitable value automatically.

### Lasso Regression

Lasso Regression applies L1 regularization.

`LassoCV` is used to select the regularization parameter through cross-validation.

The number of non-zero coefficients is also analysed to evaluate the effect of Lasso on model complexity.

### Elastic Net

Elastic Net combines L1 and L2 regularization.

Different values of `alpha` and `l1_ratio` are evaluated using `ElasticNetCV`.

This provides a comparison between pure L1 regularization, pure L2 regularization and a combination of both.

## Model Evaluation

The models are compared using several evaluation techniques.

### Mean Squared Error

MSE is used as the main metric for measuring prediction error.

Lower MSE values indicate better predictive performance.

### R² Score

R² is used to measure how much of the variance in the target variable is explained by the model.

### Cross-Validation

5-fold cross-validation is used to evaluate model performance across different subsets of the dataset.

The analysis includes:

- `cross_val_score`
- `cross_validate`
- `KFold`

The experiments are performed both with and without feature standardization.

### Model Complexity

For the regularized models, the number of non-zero coefficients is calculated.

This provides an additional way to compare predictive performance against model complexity.

## Visual Analysis

Several visualizations are used during the analysis:

- Feature correlation matrix
- Distribution of property prices
- Learning curves for Lasso and Ridge
- MSE comparison between models
- Residual distributions for Lasso, Ridge and Elastic Net

Residual plots are used to investigate how prediction errors are distributed across the predicted price range.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Project Structure

```text
.
├── ML_fundamentals_project.ipynb
├── housing.csv
└── README.md
