import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline

# Test data loading
def test_load_data():
    df = load_data()
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    pipe = create_pipeline(X_train)
    assert "preprocessor" in pipe.named_steps, "preprocessor missing in pipeline"
    assert "classifier" in pipe.named_steps, "classifier missing in pipeline"