from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
# (60000, 28, 28)

# Reshape and normalize the data
X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

X_train = X_train / 255.0  # Normalize to [0, 1]
X_test = X_test / 255.0

# Create and train the model
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")


img = Image.open('test.png')

# Convert to grayscale
img_gray = img.convert('L')

data = list(img_gray.getdata())

# Inversing the data
for i in range(len(data)):
    data[i] = 255 - data[i]

# Only value 0 and 1 to make model faster
data = np.array(data)/255

# Converts into 2D
data = data.reshape(1, -1)

# Generate confusion matrix
y_pred = clf.predict(data)

print("Prediction:",y_pred)