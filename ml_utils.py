#!/usr/bin/env python3

'''
This script contains helping functions related to training models and making predictions.
'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump, load

models = {
    "LinearRegression": LinearRegression,
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
}

parameters = {
    "DecisionTree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'splitter': ['best', 'random']
    },
    "RandomForest": {
        'n_estimators': [10, 20, 30, 40, 50, 100, 150, 200],
        'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36]
    }
}


def train_model(model_type, with_search_params, output_model, X_train, y_train, X_test, y_test):
    """
    This function trains a regresson model given train and test data.
    It takes the model type as a parameter.
    It tunes the hyper-parameter using cross-validation grid search.
    It returns a regressor object trained on the best hyper-parameters.
    """

    if with_search_params and model_type in parameters:
        search = GridSearchCV(
            estimator=models[model_type](),
            param_grid=parameters[model_type],
            cv=15,
            n_jobs=5,
        )

        print('Hyper-parameter tuning started ..')
        search.fit(X_train, y_train)

        print("Hyper-parameters tuning is complete.")

        scores = search.cv_results_['mean_test_score']
        for score, params in zip(scores, search.cv_results_['params']):
            print("Score : %0.3f , Parameters : %r" % (score, params))

        print('Best hyper-parameters : ', search.best_params_)

        if model_type in models:
            model = models[model_type](**search.best_params_)
        else:
            print("model_type is not found.")
            return
    else:
        if model_type in models:
            model = models[model_type]()
        else:
            print("model_type is not found.")
            return

    print("Training model ...")
    model.fit(X_train, y_train)
    print("Training is complete.")

    print("Saving model into " + output_model)
    dump(model, output_model)

    return model



def evaluate_model(model, X_train, y_train, X_test, y_test):
    '''
    This function calculates the r2 score on the train and test sets.
    '''
    print("Mean Squared Error on train data: ", mean_squared_error(y_train, model.predict(X_train)))
    print("Mean Squared Error on test data: ", mean_squared_error(y_test, model.predict(X_test)))
    print("R2 score on train data :", model.score(X_train, y_train))
    print("R2 score on test data :", model.score(X_test, y_test))


def predict_apartment(features, model_path):
    '''
    This function predicts the price given the necessary features.
    '''
    model = load(model_path)
    predicted_price = model.predict([features])[0]
    return predicted_price
