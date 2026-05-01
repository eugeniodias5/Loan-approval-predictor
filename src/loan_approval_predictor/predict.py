from loan_approval_predictor.processing.data_management import load_pipeline, load_data


def predict(X):
    pipeline = load_pipeline()

    return pipeline.predict(X)


if __name__ == "__main__":
    X, _ = load_data()
    predictions = predict(X)
    print(predictions)
