import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Read the dataaset
iris_df = pd.read_csv('data/iris.csv')
# Generate a random selection of rows or columns from a DataFrame
iris_df.sample(
    frac=1,  # Shuffle the entire DataFrame. Randomly selects 100% of the data
    random_state=42  # Ensure reproducibility of random sampling 
)

# Feature selection
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris_df[features]
# Target
y = iris_df['Species']

# Split data into train and test sets
# 70% training and 30% test
# Stratify ensures the resulting training and testing sets
# maintain the same proportion of classes as the original dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create an instance of the random forest classifier
# n_estimators determines the number of decision trees in the forest
model_classifier = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training data
model_classifier.fit(X_train, y_train)

# Predit on the test set
y_pred = model_classifier.predict(X_test)

# Calculate accuracy of predictions
# The proportion of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# Save the model to disk
joblib.dump(model_classifier, 'random_forest_model.sav')