# Iris Flower Classification using Support Vector Machine (SVM)

This project demonstrates the use of a Support Vector Machine (SVM) model to classify the Iris dataset into its respective classes based on different flower features.

The dataset is available through the `sklearn.datasets.load_iris()` function and is a well-known dataset used for classification tasks.

## How the SVM Model Works 

Support Vector Machines (SVMs) are a type of supervised learning algorithm that can be used for classification or regression tasks. For classification problems like this one, the goal of SVM is to find the optimal hyperplane that best separates the different classes in the feature space.

### Key Concepts of SVM:

1. Hyperplane:
A hyperplane is a decision boundary that separates the classes. In two-dimensional space, this is just a line, but in higher dimensions, it’s a plane (or a hyperplane).

2. Support Vectors:
These are the data points that are closest to the hyperplane. SVM focuses on these points because they have the most influence on the position and orientation of the hyperplane.

3. Maximal Margin:
The optimal hyperplane is the one that maximizes the margin, which is the distance between the hyperplane and the nearest support vectors from either class. By maximizing this margin, the SVM creates the most robust classifier.

4. Kernel Trick:
SVM can also use a mathematical technique called the kernel trick to handle non-linearly separable data. By mapping the input data to a higher-dimensional space, SVM can find a linear decision boundary that works even for more complex data distributions.

### Process:

1. Training Phase:
The SVM algorithm takes the training data (X_train, y_train) and tries to find the best hyperplane that separates the different classes.
The algorithm identifies the support vectors and constructs the optimal hyperplane that maximizes the margin between classes.

2. Prediction Phase:
Once the model is trained, it uses the learned hyperplane to classify the test data (X_test).
The model predicts the class label of the test data by determining on which side of the hyperplane the data points fall.

3. Evaluation:
After predictions are made, the accuracy is calculated by comparing the predicted labels to the true labels from the test set.

## Requirements

To run this code, you will need to install the following Python libraries:

- `sklearn`

You can install the required library using pip:

```bash
pip install scikit-learn
```
## Code Overview

1. Data Loading:
   - The Iris dataset is loaded using load_iris().

2. Data Splitting:
   - The dataset is split into features (X) and labels (y).
   The data is further split into training and testing sets using train_test_split().

3. Model Training:
   - A Support Vector Classifier (SVC) model is created and trained using the training data (X_train, y_train).

4. Prediction:
   - The trained model is used to make predictions on the test set (X_test).

5. Accuracy Evaluation:
   - The accuracy of the model is calculated using accuracy_score() by comparing the predicted labels with the actual labels in the test set.
