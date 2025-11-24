import joblib

model = joblib.load('titanic_model.pkl')

# prediction = model.predict([[1045, 3, 0.512725, 0, 2, -0.419333, False, False, True]])
prediction = model.predict([[963, 3, -0.675087585, 0, 0, -0.496022013, True, False, True]])
print(prediction)