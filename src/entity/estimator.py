import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts raw input dataframe, applies SAME feature engineering
        used during training, then performs scaling + prediction.
        """
        try:
            logging.info("Starting prediction process.")
    
            df = dataframe.copy()
    
            # ---------- SAME FEATURE ENGINEERING AS TRAINING ----------
    
            # Map Gender column
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
    
            # Drop id column if present
            if "id" in df.columns:
                df = df.drop("id", axis=1)
    
            # Create dummy variables
            df = pd.get_dummies(df, drop_first=True)
    
            # Rename columns exactly like training step
            df = df.rename(columns={
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
            })
    
            # Align columns with training preprocessor
            expected_cols = self.preprocessing_object.feature_names_in_
            df = df.reindex(columns=expected_cols, fill_value=0)
    
            # ---------- APPLY PREPROCESSOR ----------
            logging.info("Applying preprocessing pipeline")
            transformed_feature = self.preprocessing_object.transform(df)
    
            # ---------- MODEL PREDICTION ----------
            logging.info("Using trained model for prediction")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
          logging.error("Error occurred in predict method", exc_info=True)
          raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"