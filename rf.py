import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
import xgboost
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
import shap
warnings.filterwarnings("ignore", category=FutureWarning)

FILE_PATH = "C:\\Users\\Anthony\\Desktop\\kaggle-titanic-competition\\titanic\\"

def load_data():
    train = pd.read_csv(FILE_PATH + "train.csv")
    test = pd.read_csv(FILE_PATH + "test.csv")
    return train, test


def fill_missing_values(dataset):
    dataset['Filled_Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Filled_Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Filled_Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    return dataset

def engineer_features(dataset):
    # Extract Ticket Prefixes
    def extract_ticket_prefix(ticket):
        match = re.match(r'([A-Za-z./ ]+)', ticket)
        if match:
            return match.group(1).replace(".", "").replace("/", "").strip()
        else:
            return 'None'

    dataset['TicketPrefix'] = dataset['Ticket'].apply(extract_ticket_prefix)

    # Count Number of Cabins
    dataset['NumCabins'] = dataset['Cabin'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(' ')))

    # Create Title column
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')
    dataset['Deck'] = dataset['Cabin'].str[0]
    
    # Family related features
    dataset['IsAlone'] = ((dataset['SibSp'] + dataset['Parch']) == 0).astype(int)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'].fillna(0, inplace=True)
    
    bins = [0, 18, 35, 50, np.inf]
    labels = ['Child', 'YoungAdult', 'Adult', 'Senior']
    dataset['AgeGroup'] = pd.cut(dataset['Filled_Age'], bins, labels=labels)
    dataset['AgeGroup'] = dataset['AgeGroup'].cat.codes

    dataset['IsChild'] = (dataset['Age'] < 18).astype(int)  # Convert to 0 and 1

    dataset['Spouses'] = 0
    condition = (dataset['Age'] > 18) & (dataset['SibSp'] > 0)
    dataset.loc[condition, 'Spouses'] = dataset['SibSp']

    # Convert 'Sex' into a binary column
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})

    # One-hot encoding
    dataset = pd.get_dummies(dataset, columns=['Filled_Embarked', 'Deck', 'TicketPrefix'])

    # Feature Scaling
    scaler = StandardScaler()
    features_to_scale = ['Filled_Age', 'Filled_Fare']  
    dataset[features_to_scale] = scaler.fit_transform(dataset[features_to_scale])

    # Drop columns that are no longer needed
    dataset.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'Age', 'Cabin', 'Embarked'], inplace=True)

    return dataset

def xgb_evaluate(learning_rate, n_estimators, max_depth, min_child_weight, subsample,
                 colsample_bytree, colsample_bylevel, gamma, reg_alpha, reg_lambda, scale_pos_weight):
    
    params = {
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'scale_pos_weight': scale_pos_weight
    }
    
    clf = XGBClassifier(**params)
    
    # Implementing 3-fold cross validation here
    cross_val_scores = cross_val_score(clf, features, labels, cv=3, scoring='accuracy')
    
    return cross_val_scores.mean()

train, test = load_data()

train, test = fill_missing_values(train), fill_missing_values(test)
train, test = engineer_features(train), engineer_features(test)

# Ensure both train and test datasets have the same set of columns
train, test = train.align(test, join='left', axis=1)
test = test.fillna(0)

# Assuming target column is 'Survived' in the train dataset (common for Titanic datasets)
features = train.drop(['PassengerId', 'Survived'], axis=1)
labels = train['Survived']


bounds = {
    'learning_rate': (0.05, 0.5),
    'n_estimators': (50, 150),
    'max_depth': (4, 6),
    'min_child_weight': (1, 5),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'colsample_bylevel': (0.6, 1.0),
    'gamma': (0.5, 1.0),
    'reg_alpha': (1.0, 2.0),
    'reg_lambda': (2.0, 3.0),
    'scale_pos_weight': (1, 4)
}

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=bounds,
    random_state=1
)

optimizer.maximize(init_points=10, n_iter=10)

# After optimization, print the best parameters
print(optimizer.max)

# Train the model using the best parameters on the entire training set
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

# Assuming target column is 'Survived' in the train dataset (common for Titanic datasets)
features = train.drop(['PassengerId', 'Survived'], axis=1)
labels = train['Survived']

model = XGBClassifier(**best_params)
model.fit(features, labels)

# Make predictions
test_features = test[features.columns]
predictions = model.predict(test_features)

# Create submission DataFrame and save it
submission_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission_file_path = FILE_PATH + "submission.csv"
submission_df.to_csv(submission_file_path, index=False)

print("Submission file saved to", submission_file_path)

# After fitting the model
importances = model.feature_importances_
feature_names = features.columns

# Pair the feature names with their importance scores
feature_importances = list(zip(feature_names, importances))

# Sort the feature importances by score in descending order
sorted_feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print out all the features in the sorted order of importance
print("Features in order of importance:")
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance}")


# Initialize JS visualization code (needed for plotting SHAP values later)
shap.initjs()

# Create a SHAP explainer object
explainer = shap.TreeExplainer(model)

# Convert the sample data into DMatrix with the enable_categorical set to True
sample_dmatrix = xgboost.DMatrix(features.iloc[0:1, :], enable_categorical=True)

# Now, compute the SHAP values for the sample DMatrix
shap_values = explainer.shap_values(sample_dmatrix)

# Visualize the prediction's explanation
shap.force_plot(explainer.expected_value, shap_values, features.iloc[0])