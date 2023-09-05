import os
import shutil
import tensorflow as tf
import pandas as pd
import tabnet

from sklearn.model_selection import train_test_split

# Constants
BATCH_SIZE = 500
num_classes = 2  # Survived or not

# Load datasets
train_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\train.csv")
test_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\test.csv")

# Extract labels
train_labels = train_dataset['Survived']
train_dataset.drop(columns=['Survived'], inplace=True) 

# Handle missing values
train_dataset['Filled_Age'] = train_dataset['Age'].fillna(train_dataset['Age'].median())
train_dataset['Filled_Embarked'] = train_dataset['Embarked'].fillna(train_dataset['Embarked'].mode()[0])

test_dataset['Filled_Age'] = test_dataset['Age'].fillna(test_dataset['Age'].median())
test_dataset['Filled_Fare'] = test_dataset['Fare'].fillna(test_dataset['Fare'].median())
test_dataset['Filled_Embarked'] = test_dataset['Embarked'].fillna(test_dataset['Embarked'].mode()[0])

# Additional feature engineering for 'Children' and 'Spouses'
train_dataset['Children'] = 0
train_dataset.loc[(train_dataset['Sex'] == 'female') & (train_dataset['Parch'] > 0), 'Children'] = train_dataset['Parch']

test_dataset['Children'] = 0
test_dataset.loc[(test_dataset['Sex'] == 'female') & (test_dataset['Parch'] > 0), 'Children'] = test_dataset['Parch']

train_dataset['Spouses'] = 0
train_dataset.loc[train_dataset['Sex'] == 'male', 'Spouses'] = train_dataset['SibSp']

test_dataset['Spouses'] = 0
test_dataset.loc[test_dataset['Sex'] == 'male', 'Spouses'] = test_dataset['SibSp']

# Define the features for the model
selected_features = ['Pclass', 'Sex', 'Filled_Age', 'Children', 'Spouses', 'Fare', 'Filled_Embarked']


# Preprocess the Titanic dataset
def preprocess_data(features, age_median, fare_median, embarked_mode):
    # Normalize continuous features
    features['Filled_Age'] = features['Age'].fillna(age_median)
    features['Filled_Fare'] = features['Fare'].fillna(fare_median)
    features['Filled_Embarked'] = features['Embarked'].fillna(embarked_mode)
    
    # Drop the original columns with NaN values
    features.drop(columns=['Age', 'Cabin', 'Embarked'], inplace=True)

    # One-hot encode categorical features
    features = pd.get_dummies(features, columns=['Sex', 'Filled_Embarked', 'Pclass'])

    return features

# Get median and mode values from the training set
age_median = train_dataset['Age'].median()
fare_median = train_dataset['Fare'].median()
embarked_mode = train_dataset['Embarked'].mode()[0]

# Preprocess the datasets
train_features = preprocess_data(train_dataset.copy(), age_median, fare_median, embarked_mode)
test_features = preprocess_data(test_dataset.copy(), age_median, fare_median, embarked_mode)

# Identify and drop non-numeric columns
non_numeric_columns = train_features.select_dtypes(exclude=['number']).columns
train_features = train_features.drop(columns=non_numeric_columns)
test_features = test_features.drop(columns=non_numeric_columns)

train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)


# Convert the labels and features to float32
train_features = train_features.astype('float32')
val_features = val_features.astype('float32')
train_labels = train_labels.astype('float32')
val_labels = val_labels.astype('float32')

train_labels = train_labels.values.reshape(-1, 1)
val_labels = val_labels.values.reshape(-1, 1)


# Adjust the model configuration
num_features = train_features.shape[1] 

model = tabnet.TabNetClassifier(feature_columns=None, num_classes=1, num_features=num_features,
                                feature_dim=32, output_dim=16,
                                num_decision_steps=5, relaxation_factor=1.5,
                                sparsity_coefficient=0., batch_momentum=0.98,
                                virtual_batch_size=None, norm_type='group',
                                num_groups=-1)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=15, validation_data=(val_features, val_labels), batch_size=BATCH_SIZE, verbose=2)

predictions = model.predict(test_features)

submission_data = pd.DataFrame({
    'Not_Survived': (1 - predictions).squeeze(),   # (1 - predictions) gives the probability of Not_Survived
    'Survived': predictions.squeeze()              # .squeeze() is used to flatten the array into 1D
})

# Save predictions to CSV
predictions = model.predict(test_features)
submission_data = pd.DataFrame(predictions, columns=['Not_Survived', 'Survived'])
submission_data.to_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\tensorflow_predictions.csv", index=False)
