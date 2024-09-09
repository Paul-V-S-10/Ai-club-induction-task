import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Sex']),
        ('num', StandardScaler(), ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight'])
    ])

# Define pipeline with SVR
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])

# Separate features and target
X_train = train_data.drop(['id', 'Age'], axis=1)
y_train = train_data['Age']

# Fit and evaluate using cross-validation
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=mae_scorer)

# Print cross-validated MAE
print(f'Cross-validated MAE: {-cv_scores.mean()}')

# Fit the model on the full training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
X_test = test_data.drop('id', axis=1)
y_pred_test = pipeline.predict(X_test)

# Create a submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'Age': y_pred_test
})

submission.to_csv('submission.csv', index=False)
