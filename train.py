import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load datasets
train_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\train.csv")
test_dataset = pd.read_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\test.csv")

deck_encoder = LabelEncoder()

# Handle missing values
train_dataset['Filled_Age'] = train_dataset['Age'].fillna(train_dataset['Age'].median())
train_dataset['Filled_Embarked'] = train_dataset['Embarked'].fillna(train_dataset['Embarked'].mode()[0])

test_dataset['Filled_Age'] = test_dataset['Age'].fillna(test_dataset['Age'].median())
test_dataset['Filled_Fare'] = test_dataset['Fare'].fillna(test_dataset['Fare'].median())
test_dataset['Filled_Embarked'] = test_dataset['Embarked'].fillna(test_dataset['Embarked'].mode()[0])

# Convert categorical variables to numeric using label encoding
label_encoders_dict = {}
for column_name in ['Sex', 'Filled_Embarked']:
    encoder = LabelEncoder()
    train_dataset[column_name + '_Encoded'] = encoder.fit_transform(train_dataset[column_name])
    test_dataset[column_name + '_Encoded'] = encoder.transform(test_dataset[column_name])
    label_encoders_dict[column_name] = encoder

# Feature engineering

# Family Size
train_dataset['FamilySize'] = train_dataset['SibSp'] + train_dataset['Parch']
test_dataset['FamilySize'] = test_dataset['SibSp'] + test_dataset['Parch']

# Alone or Not
train_dataset['IsAlone'] = 0
train_dataset.loc[train_dataset['FamilySize'] == 0, 'IsAlone'] = 1

test_dataset['IsAlone'] = 0
test_dataset.loc[test_dataset['FamilySize'] == 0, 'IsAlone'] = 1

# Cabin Information (Deck)
train_dataset['Deck'] = train_dataset['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
test_dataset['Deck'] = test_dataset['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')

train_dataset['Deck_Encoded'] = deck_encoder.fit_transform(train_dataset['Deck'])
test_dataset['Deck_Encoded'] = deck_encoder.transform(test_dataset['Deck'])
train_dataset['IsChild'] = 0
train_dataset.loc[train_dataset['Filled_Age'] < 18, 'IsChild'] = 1

test_dataset['IsChild'] = 0
test_dataset.loc[test_dataset['Filled_Age'] < 18, 'IsChild'] = 1

# Identifying mothers
train_dataset['IsMother'] = 0
train_dataset.loc[(train_dataset['Sex'] == 'female') & (train_dataset['Filled_Age'] > 18) & (train_dataset['Parch'] > 0), 'IsMother'] = 1

test_dataset['IsMother'] = 0
test_dataset.loc[(test_dataset['Sex'] == 'female') & (test_dataset['Filled_Age'] > 18) & (test_dataset['Parch'] > 0), 'IsMother'] = 1

# Identifying spouses for males (assuming the 'SibSp' column for males refers to spouses)
train_dataset['Spouses'] = 0
train_dataset.loc[train_dataset['Sex'] == 'male', 'Spouses'] = train_dataset['SibSp']

test_dataset['Spouses'] = 0
test_dataset.loc[test_dataset['Sex'] == 'male', 'Spouses'] = test_dataset['SibSp']

# Define the features for the model
selected_features = ['Pclass', 'Sex_Encoded', 'Filled_Age', 'IsChild', 'IsMother', 'Spouses', 'Fare', 'Filled_Embarked_Encoded']

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_dataset[selected_features])
test_features = scaler.transform(test_dataset[selected_features])

train_labels = train_dataset['Survived'].values

# Define and compile the model in a function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(len(selected_features),)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Implement k-fold cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_num = 1
for train_index, val_index in kfold.split(train_features, train_labels):
    model = create_model()
    print(f"Training on fold {fold_num}/{num_folds}")
    model.fit(train_features[train_index], train_labels[train_index], epochs=25, validation_data=(train_features[val_index], train_labels[val_index]), verbose=2)
    fold_num += 1

# Train on the entire training set
model = create_model()
model.fit(train_features, train_labels, epochs=25, verbose=1)

# Make predictions on the test dataset
test_dataset_predictions = (model.predict(test_features) > 0.5).astype("int32")

# Prepare the results for submission
submission_data = pd.DataFrame({
    'PassengerId': test_dataset['PassengerId'],
    'Survived': test_dataset_predictions[:, 0]
})

# Save the predictions to a CSV file
submission_data.to_csv("C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\tensorflow_predictions.csv", index=False)