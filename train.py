# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

# Hyperparameters
BATCH_SIZE = 891
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.00001
EPOCHS = 35

USE_VALIDATION_SPLIT = True

FILE_PATH = "C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\"

def load_data():
    train = pd.read_csv(FILE_PATH + "train.csv")
    test = pd.read_csv(FILE_PATH + "test.csv")
    return train, test

def fill_missing_values(dataset):
    dataset['Filled_Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Filled_Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Filled_Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

def engineer_features(dataset):
    # Create Title column
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')
    dataset['Deck'] = dataset['Cabin'].str[0]
    
    # Family related features
    dataset['IsAlone'] = ((dataset['SibSp'] + dataset['Parch']) == 0).astype(int)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)

    bins = [0, 18, 35, 50, np.inf]
    labels = ['Child', 'YoungAdult', 'Adult', 'Senior']
    dataset['AgeGroup'] = pd.cut(dataset['Filled_Age'], bins, labels=labels)

    # Add IsChild feature
    dataset['IsChild'] = dataset['Age'] < 18
    dataset['IsChild'] = dataset['IsChild'].astype(int)  # Convert to 0 and 1

    # Add IsMother feature
    dataset['IsMother'] = 0
    condition = (dataset['Sex'] == 'female') & (dataset['Age'] > 18) & (dataset['SibSp'] >= 1) & (~dataset['Name'].str.contains('Miss.'))
    dataset.loc[condition, 'IsMother'] = 1

    # Add Spouses feature
    dataset['Spouses'] = 0
    condition = (dataset['Age'] > 18) & (dataset['SibSp'] > 0)
    dataset.loc[condition, 'Spouses'] = dataset['SibSp']

    dataset['Title'].fillna(0, inplace=True)

    # Fill Fare with median value
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

    dataset.drop(columns=['Name', 'Age', 'Embarked', 'Cabin', 'Ticket', 'SibSp', 'Parch'], inplace=True)

def add_model_predictions_as_features(train_dataset, test_dataset, numerical_features, categorical_features):
    # Prepare data for prediction
    features_train = [train_dataset[feature].values for feature in numerical_features + categorical_features]
    features_test = [test_dataset[feature].values for feature in numerical_features + categorical_features]
    
    # Train Random Forest model
    features_train_stacked = np.vstack(features_train).T
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(features_train_stacked, train_dataset['Survived'])

    # Predict with Random Forest
    train_dataset['RF_Predictions'] = rf_model.predict(features_train_stacked)
    test_dataset['RF_Predictions'] = rf_model.predict(np.hstack(features_test).reshape(-1, len(numerical_features + categorical_features)))
    
    # Train Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(features_train_stacked, train_dataset['Survived'])
    
    # Predict with Naive Bayes
    train_dataset['NB_Predictions'] = nb_model.predict(features_train_stacked)
    test_dataset['NB_Predictions'] = nb_model.predict(np.hstack(features_test).reshape(-1, len(numerical_features + categorical_features)))
    
    # Add the new features to the numerical features list
    numerical_features.extend(['RF_Predictions', 'NB_Predictions'])

def encode_features(train_dataset, test_dataset, categorical_features):
    for feature in categorical_features:
        train_dataset[feature] = train_dataset[feature].astype('category').cat.codes
        test_dataset[feature] = test_dataset[feature].astype('category').cat.codes

def normalize_features(train_dataset, test_dataset, numerical_features):
    scaler = MinMaxScaler()
    train_dataset[numerical_features] = scaler.fit_transform(train_dataset[numerical_features])
    test_dataset[numerical_features] = scaler.transform(test_dataset[numerical_features])

def split_data(train_cat_data, train_numerical_features, train_labels):
    if USE_VALIDATION_SPLIT:
        features_training_num, features_validation_num, labels_training, labels_validation = train_test_split(
            train_numerical_features, train_labels, test_size=0.2, random_state=42)

        features_training_cat = [train_test_split(cat_feature, test_size=0.2, random_state=42)[0] for cat_feature in train_cat_data]
        features_validation_cat = [train_test_split(cat_feature, test_size=0.2, random_state=42)[1] for cat_feature in train_cat_data]

        features_training = [features_training_num] + features_training_cat
        features_validation = [features_validation_num] + features_validation_cat

        return features_training, labels_training, features_validation, labels_validation
    else:
        return [train_numerical_features] + train_cat_data, train_labels, None, None

def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def create_model(input_shape):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Load and preprocess data
    train_dataset, test_dataset = load_data()
    fill_missing_values(train_dataset)
    fill_missing_values(test_dataset)
    engineer_features(train_dataset)
    engineer_features(test_dataset)

    categorical_features = ['Pclass', 'Sex', 'Filled_Embarked', 'Deck', 'AgeGroup', 'Title']
    numerical_features = ['IsChild', 'IsMother', 'Spouses', 'Fare']

    encode_features(train_dataset, test_dataset, categorical_features)
    normalize_features(train_dataset, test_dataset, numerical_features)
    add_model_predictions_as_features(train_dataset, test_dataset, numerical_features, categorical_features)

    all_features = numerical_features + categorical_features
    train_features = train_dataset[all_features].values
    test_features = test_dataset[all_features].values
    train_labels = train_dataset['Survived'].values

    if USE_VALIDATION_SPLIT:
        features_training, features_validation, labels_training, labels_validation = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=42)
    else:
        features_training, labels_training = train_features, train_labels
        features_validation, labels_validation = None, None

    neural_network_model = create_model((features_training.shape[1],))

    if USE_VALIDATION_SPLIT:
        history = neural_network_model.fit(features_training, labels_training, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(features_validation, labels_validation))
    else:
        history = neural_network_model.fit(features_training, labels_training, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    plot_training_history(history)

    predictions = neural_network_model.predict(test_features)
    predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    submission_df = pd.DataFrame({'PassengerId': test_dataset['PassengerId'], 'Survived': predictions})
    submission_file_path = FILE_PATH + "submission.csv"
    submission_df.to_csv(submission_file_path, index=False)
    print("Submission file saved to", submission_file_path)

if __name__ == "__main__":
    main()
