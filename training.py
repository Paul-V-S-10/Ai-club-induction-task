import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Sex']),
        ('num', StandardScaler(), ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight'])
    ])

X_train = preprocessor.fit_transform(train_data.drop(['id', 'Age'], axis=1))
y_train = train_data['Age']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
print(f'Training MAE: {mean_absolute_error(y_train, y_pred_train)}')

X_test = preprocessor.transform(test_data.drop('id', axis=1))
y_pred_test = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_data['id'],
    'Age': y_pred_test
})

submission.to_csv('submission.csv', index=False)