import joblib


def predict(data):
    model_classifier = joblib.load('random_forest_model.sav')
    return model_classifier.predict(data)