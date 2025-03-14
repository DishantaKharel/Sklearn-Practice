# KNN-Model
# K-Nearest Neighbors (KNN) Classification

This project uses the K-Nearest Neighbors (KNN) algorithm to classify cars based on their attributes like buying price, maintenance cost, and safety rating. It uses the car evaluation dataset (`car.data`).

## What the Code Does:

1. Loads the car dataset (`car.data`).
2. Preprocesses the data by converting categorical values to numerical labels using `LabelEncoder`.
3. Splits the data into training and testing sets (80% training, 20% testing).
4. Trains a KNN classifier with 25 neighbors using the training data.
5. Makes predictions on the test data.
6. Prints the prediction results and the accuracy of the model.
