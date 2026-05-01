from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from processing.preprocessors import GaussianTransformer, StandardScaler

from config.config import LOG_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_SEED


def _get_preprocessor():
    preprocessor = ColumnTransformer(transformers=[
        ("log", GaussianTransformer(method='yeo-johnson'), LOG_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])
    return preprocessor


def get_pipeline(model_type):
    if model_type == "logistic":
        classifier = LogisticRegression(random_state=RANDOM_SEED)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(random_state=RANDOM_SEED)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    preprocessor = _get_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline


def get_voting_pipeline(models):
    voting_pipeline = Pipeline([
        ("classifier", VotingClassifier(estimators=models, voting='soft'))
    ])
    return voting_pipeline
