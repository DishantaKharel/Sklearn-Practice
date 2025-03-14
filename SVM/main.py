from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#Split it in features and lables
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train,y_train)

print(model)

prediction = model.predict(X_test)
acc = accuracy_score(y_test, prediction)

print("Prediction:", prediction)
print("actual:", y_test)
print("Accuracy:", acc)

