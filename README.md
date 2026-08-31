# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and functions.
2. Load the Iris dataset using load_iris().
3. Separate the input features X and target variable y.
4. Split the dataset into training and testing sets using train_test_split().
5. Standardize the input features using StandardScaler.
6. Create the SGDClassifier model using log_loss.
7. Train the model using the training data.
8. Predict the Iris species using the testing data.
9. Convert the predicted class numbers into species names.
10. Calculate the accuracy using accuracy_score().
11. Generate the confusion matrix.
12. Display the predicted species, actual species, accuracy, and confusion matrix.


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Muruga s
RegisterNumber:  212225040265
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Input features
X = iris.data

# Target variable
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Classifier
model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict the species
y_pred = model.predict(X_test)

# Convert predicted numbers to species names
predicted_species = iris.target_names[y_pred]
actual_species = iris.target_names[y_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predicted Iris Species:")
print(predicted_species)

print("\nActual Iris Species:")
print(actual_species)

print("\nAccuracy:", accuracy)

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

## Output:
<img width="905" height="447" alt="image" src="https://github.com/user-attachments/assets/80e170bf-6a59-4510-8747-5775e537855e" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
