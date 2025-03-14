# California Housing Price Prediction using Linear Regression

This project demonstrates the use of a **Linear Regression** model to predict housing prices in California based on the California Housing dataset. The dataset is available through the `sklearn.datasets.fetch_california_housing()` function.

## Requirements

To run this code, you will need to install the following Python libraries:

- `sklearn`
- `matplotlib`
- `numpy`
- `pandas`

You can install the required libraries using pip:

```bash
pip install scikit-learn matplotlib numpy pandas
```

## Code Overview
The code performs the following steps:

1. Data Loading:
   - The California housing dataset is loaded using fetch_california_housing().

2. Data Splitting:
   - The dataset is split into features (X) and target values (y). The data is further split into training and testing sets using train_test_split().

3. Data Visualization:
   - A scatter plot is created to show the relationship between one feature (X.T[2]) and the target housing prices.

4. Model Training:
   - A Linear Regression model is created using LinearRegression(), and the model is trained using the training data (X_train, y_train).

5. Prediction:
   - The trained model is used to make predictions on the test set (X_test).

6. Accuracy Evaluation:
   - The accuracy of the model is calculated using the R² value by calling score().

7. Model Insights:
   - The model’s coefficients and intercept are printed to understand how each feature affects the housing price prediction.

## How the Linear Regression Model Works
Linear regression is one of the simplest algorithms for predicting a continuous target value based on input features. The algorithm assumes a linear relationship between the input features and the target variable.

### Key Concepts of Linear Regression:
1. Linear Relationship:

    The model assumes that the relationship between the features and the target can be described as a straight line (or hyperplane in higher dimensions).

2. Goal:

    The goal of the linear regression model is to find the best-fit line (or hyperplane) that minimizes the error between the predicted and actual target values. This is typically done by minimizing the sum of squared residuals (errors).

### Process:
1. Training Phase:
The model is trained on the training data by finding the best-fitting coefficients and intercept that minimize the prediction error.

2. Prediction Phase:
Once the model is trained, it predicts the target values for the test data based on the learned coefficients and intercept.

3. Evaluation:
The R² value is used to evaluate how well the model's predictions match the actual target values. An R² value closer to 1 indicates better model performance.

### Output
The output of the code includes:

A scatter plot showing the relationship between one feature and the target housing prices.

The predicted housing prices on the test set.

The actual housing prices in the test set.

The R² value, indicating the accuracy of the model.

The coefficients and intercept, which show how each feature affects the prediction.