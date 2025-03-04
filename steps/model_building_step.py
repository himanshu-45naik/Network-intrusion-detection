from zenml import step
import pandas as pd
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from models.logistic_regression import LogisticModelBuilder, LogisticRegressionModel
from models.svm_model import SvcModelBuilder, SvmModel
from models.oc_svm import OCsvmModelBuilder, OneClassSvmModel
from models.randomforest import RandomForestModelBuilder, RandomForestModel
from models.lgbm_model import LightGBMBuilder, LightGBMModel
from models.xgboost_model import Xgbbuilder, XgbModel
from sklearn.pipeline import Pipeline


@step(output_materializers={"output": SklearnMaterializer})
def model_building(X_train: pd.DataFrame, y_train: pd.Series, model_name: str)-> Pipeline:
    """Builds and trains the model

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training data.
        Model (str): The model to be trained
    """
    if model_name == "logisticregression":
        model = LogisticModelBuilder(LogisticRegressionModel())
    elif model_name == "svm":
        model = SvcModelBuilder(SvmModel())
    elif model_name == "ocsvm":
        model = OCsvmModelBuilder(OneClassSvmModel())
    elif model_name == "rf_binary":
        model = RandomForestModelBuilder(RandomForestModel())
    elif model_name == "rf_multiclass":
        model = RandomForestModelBuilder(RandomForestModel())
    elif model_name == "xgb_binary":
        model = Xgbbuilder(XgbModel(binary_class=True))
    elif model_name == "xgb_multiclass":
        model = Xgbbuilder(XgbModel(binary_class=False))
    elif model_name == "lgbm_binary":
        model = Xgbbuilder(XgbModel(binary_class=True))
    elif model_name == "lbgm_multiclass":
        model = Xgbbuilder(XgbModel(binary_class=False))
    
    best_model = model.execute_strategy(X_train, y_train)

    return best_model

if __name__ == "__main__":
    pass