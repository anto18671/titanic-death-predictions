import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load the data
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')

# Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

train_data['IsAlone'] = 0
train_data.loc[train_data['FamilySize'] == 0, 'IsAlone'] = 1

test_data['IsAlone'] = 0
test_data.loc[test_data['FamilySize'] == 0, 'IsAlone'] = 1

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
train_data['Title'] = train_data['Title'].replace(rare_titles, 'Rare')
test_data['Title'] = test_data['Title'].replace(rare_titles, 'Rare')

title_mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
train_data['Title'] = train_data['Title'].replace(title_mapping)
test_data['Title'] = test_data['Title'].replace(title_mapping)

train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

# Preprocessing
num_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
cat_features = ['Sex', 'Embarked', 'Title', 'IsAlone']

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

X = preprocessor.fit_transform(train_data.drop('Survived', axis=1))
y = train_data['Survived'].values

# Model Training with LightGBM
lgb_classifier = lgb.LGBMClassifier(random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100]
}

grid_search = GridSearchCV(lgb_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# If you wish, you can then train the model with the best parameters on the entire training data and make predictions on the test data.
