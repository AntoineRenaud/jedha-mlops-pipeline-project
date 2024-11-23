import mlflow
import os

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from mlflow.models import infer_signature

import data_processing

def load_data(filepath):    
    if not os.path.exists(filepath):
        data_processing.processing()
    df = pd.read_csv(filepath)   
    return df        

def preprocess_data(df): 
    X = df.drop(['cc_num','zip','trans_num','unix_time','is_fraud'], axis=1)
    y = df['is_fraud']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_pipeline(X):
    numeric_features = []
    categorical_features = []

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]) and not pd.api.types.is_bool_dtype(X[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    print("Numeric features : ", numeric_features)
    print("Categorical features : ", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder",OneHotEncoder(drop='first'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier())
        ]
    )
    print('Pipeline created.')
    return pipe  

def train_model(pipe,X_train,y_train, param_grid, cv=3, verbose=3):
    model = GridSearchCV(pipe, param_grid, cv=cv, verbose=verbose, scoring='f1')
    print('Model training...')
    model.fit(X_train, y_train)
    print('done.')
    return model

# Entry point for the script
if __name__ == "__main__":
    # Define experiment parameters
    
    EXPERIMENT_NAME="mlops-project"
    mlflow.set_tracking_uri(os.environ["APP_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    artifact_path = "fraud_detection"
    registered_model_name = "fraud_detection"

    filepath = 'src/cleaned_dataset.csv'
    param_grid = {
    "classifier__max_depth": [1, 3, 6],
    "classifier__min_child_weight": [3,5,7,9],
    "classifier__n_estimators": [10,20,50],
    }

    df = load_data(filepath)
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    pipe = create_pipeline(X_train)
    model = train_model(pipe,X_train,y_train,param_grid,cv=3,verbose=2)
    print("...Done.") 
    
with mlflow.start_run(experiment_id = experiment.experiment_id, run_name='XGBClassifier'):

    mlflow.log_param("classifier", "XGBClassifier")
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("encoder", "OneHotEncoder(drop='first')")
        
    print("f1 score on training set : ", model.score(X_train, y_train))
    print("f1 score on test set : ", model.score(X_test, y_test))

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model, artifact_path=artifact_path, registered_model_name=registered_model_name)

    #F1 score Metric 
    f1score = f1_score(y_test, y_test_pred)
    print("F1 score on test set :", f1score)

    #Precision Metric
    precision = precision_score(y_test, y_test_pred)
    print("Precision on test set :", precision)  

    #Recall Metric
    recall = recall_score(y_test, y_test_pred)
    print("Recall on test set :", f1score)
        
    #Log to mlflow
    mlflow.log_metric('f1_score', f1score)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)