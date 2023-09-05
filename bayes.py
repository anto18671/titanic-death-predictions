import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load datasets
train_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\train.csv")
test_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\test.csv")

# Handle missing values
train_dataset['Age'].fillna(train_dataset['Age'].median(), inplace=True)
train_dataset['Embarked'].fillna(train_dataset['Embarked'].mode()[0], inplace=True)

test_dataset['Age'].fillna(test_dataset['Age'].median(), inplace=True)
test_dataset['Fare'].fillna(test_dataset['Fare'].median(), inplace=True)
test_dataset['Embarked'].fillna(test_dataset['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric using label encoding
label_encoders_dict = {}
for column_name in ['Sex', 'Embarked']:
    encoder = LabelEncoder()
    train_dataset[column_name] = encoder.fit_transform(train_dataset[column_name])
    test_dataset[column_name] = encoder.transform(test_dataset[column_name])
    label_encoders_dict[column_name] = encoder

# Additional feature engineering for 'Children' and 'Spouses'
train_dataset['Children'] = 0
train_dataset.loc[(train_dataset['Sex'] == 0) & (train_dataset['Parch'] > 0), 'Children'] = train_dataset['Parch']

test_dataset['Children'] = 0
test_dataset.loc[(test_dataset['Sex'] == 0) & (test_dataset['Parch'] > 0), 'Children'] = test_dataset['Parch']

train_dataset['Spouses'] = 0
train_dataset.loc[train_dataset['Sex'] == 1, 'Spouses'] = train_dataset['SibSp']

test_dataset['Spouses'] = 0
test_dataset.loc[test_dataset['Sex'] == 1, 'Spouses'] = test_dataset['SibSp']

# Define the features for the model
selected_features = ['Pclass', 'Sex', 'Age', 'Children', 'Spouses', 'Fare', 'Embarked']
train_features = train_dataset[selected_features]
train_labels = train_dataset['Survived']

# Split training data for validation
train_split_features, validation_features, train_split_labels, validation_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
naive_bayes_model = GaussianNB()

# Train the model
naive_bayes_model.fit(train_split_features, train_split_labels)

# Validate the model
validation_predictions = naive_bayes_model.predict(validation_features)
validation_accuracy = accuracy_score(validation_labels, validation_predictions)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# Make predictions on the test dataset
test_dataset_predictions = naive_bayes_model.predict(test_dataset[selected_features])

# Prepare the results for submission
submission_data = pd.DataFrame({
    'PassengerId': test_dataset['PassengerId'],
    'Survived': test_dataset_predictions
})

# Save the predictions to a CSV file
submission_data.to_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\naive_bayes_predictions.csv", index=False)
