from sklearn.model_selection import train_test_split, GridSearchCV


from loan_approval_predictor.config.config import TRAIN_RATIO, MODELS
from loan_approval_predictor.pipeline import get_pipeline, get_voting_pipeline
from loan_approval_predictor.processing.data_management import load_data, save_pipeline

from loan_approval_predictor.evaluate import evaluate_model

from loan_approval_predictor.config.config import RANDOM_SEED


def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_RATIO, random_state=RANDOM_SEED
    )

    models_pipeline = {}

    for model_name in MODELS.keys():
        pipeline = get_pipeline(model_name)
        param_grid = MODELS[model_name]
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1_macro"
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        models_pipeline[model_name] = best_model
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Geting the best models for the voting classifier
    best_models = [(name, model) for name, model in models_pipeline.items()]
    voting_pipeline = get_voting_pipeline(best_models)
    voting_pipeline.fit(X_train, y_train)

    evaluate_model(voting_pipeline, X_test, y_test)
    print("Saving the pipeline...")
    save_pipeline(voting_pipeline)

    return voting_pipeline


if __name__ == "__main__":
    train_model()
