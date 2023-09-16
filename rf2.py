import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# Age Prediction
def predict_age(data):
    # Split the data
    age_train_data = data.dropna(subset=['Age'])
    age_predict_data = data[data['Age'].isnull()]

    # Prepare features for the model
    features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex', 'Embarked', 'Title']

    X_age = age_train_data[features]
    y_age = age_train_data['Age']
    
    # Splitting into training and validation set for assessing age prediction accuracy
    X_age_train, X_age_val, y_age_train, y_age_val = train_test_split(X_age, y_age, test_size=0.2, random_state=42)
    
    # Preprocess features
    preprocessor_age = ColumnTransformer(
        transformers=[
            ('num', num_transformer, ['SibSp', 'Parch', 'Fare', 'FamilySize']),
            ('cat', cat_transformer, ['Sex', 'Embarked', 'Title'])
        ])

    X_age_train_transformed = preprocessor_age.fit_transform(X_age_train)
    X_age_val_transformed = preprocessor_age.transform(X_age_val)
    X_age_predict_transformed = preprocessor_age.transform(age_predict_data[features])

    # Train the regression model
    lgb_regressor = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_regressor.fit(X_age_train_transformed, y_age_train)

    # Predict ages for validation set
    val_predictions = lgb_regressor.predict(X_age_val_transformed)
    print("Mean Absolute Error for Age Prediction:", mean_absolute_error(y_age_val, val_predictions))

    # Predict ages for the missing values
    predicted_ages = lgb_regressor.predict(X_age_predict_transformed)
    return predicted_ages

# Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
train_data['Title'] = train_data['Title'].replace(rare_titles, 'Rare')
test_data['Title'] = test_data['Title'].replace(rare_titles, 'Rare')

title_mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
train_data['Title'] = train_data['Title'].replace(title_mapping)
test_data['Title'] = test_data['Title'].replace(title_mapping)

train_data['FareBin'] = pd.cut(train_data['Fare'], bins=[0, 7.91, 14.45, 31, 512], labels=[1, 2, 3, 4], include_lowest=True)
test_data['FareBin'] = pd.cut(test_data['Fare'], bins=[0, 7.91, 14.45, 31, 512], labels=[1, 2, 3, 4], include_lowest=True)

train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

# Preprocessing
num_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FareBin']
cat_features = ['Sex', 'Embarked', 'Title']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

train_data.loc[train_data['Age'].isnull(), 'Age'] = predict_age(train_data)
test_data.loc[test_data['Age'].isnull(), 'Age'] = predict_age(test_data)

X = preprocessor.fit_transform(train_data.drop('Survived', axis=1))
y = train_data['Survived'].values

# Model Training with LightGBM
lgb_classifier = lgb.LGBMClassifier(random_state=42, silent=True)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [175],
    'max_depth': [10],
    'learning_rate': [0.05],
    'num_leaves': [30],
    'reg_alpha': [1.0],
    'reg_lambda': [0.05, 0.1, 0.2],
    'min_split_gain': [0.2],
    'min_child_samples': [25],
    'min_child_weight': [0.001],
    'verbosity': [-1]
}

grid_search = GridSearchCV(lgb_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# After your GridSearchCV fit
# 1. Get the feature importances from the best estimator
importances = grid_search.best_estimator_.feature_importances_

# 2. Extract feature names from the preprocessor
# Numerical features remain the same
num_features_names = num_features

# For categorical features, we need to extract the names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat']['onehot']
cat_features_names = cat_encoder.get_feature_names_out(cat_features)

# Combine all feature names
all_feature_names = num_features_names + list(cat_features_names)

# 3. Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})

# Sort the DataFrame based on importance values
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print sorted feature importances
print(sorted_feature_importance_df)

# 4. Save the predictions to submission.csv
X_test = preprocessor.transform(test_data)
predictions = grid_search.predict(X_test)

submission_df = pd.DataFrame({
    'PassengerId': pd.read_csv('titanic/test.csv')['PassengerId'],
    'Survived': predictions
})
submission_path = 'titanic/submission.csv'
submission_df.to_csv(submission_path, index=False)
