from sklearn.metrics import classification_report
import mlflow


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics(
        {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
        }
    )
    print(classification_report(y_test, y_pred))
