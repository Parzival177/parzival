from sklearn import datasets
from sklearn import svm

# Load the iris dataset
iris = datasets.load_iris()

# Create a support vector machine classifier
clf = svm.SVC()

# Train the classifier with the iris dataset
clf.fit(iris.data, iris.target)

# Predict the species of a new iris flower
new_flower = [[5.0, 3.6, 1.3, 0.25]]  # sepal length, sepal width, petal length, petal width
predicted_species = clf.predict(new_flower)

# Print the predicted species
print(predicted_species)
