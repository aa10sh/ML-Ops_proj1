import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer using schema-driven configuration
        """
        try:
            logging.info("Creating data transformer object")

            numerical_features = self._schema_config["num_features"]
            categorical_features = self._schema_config["categorical_columns"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_features),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_features,
                    ),
                ]
            )

            logging.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes complete data transformation pipeline
        """
        try:
            logging.info("Starting Data Transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info("Train and Test data loaded")

            # ================================
            # Select only schema-required cols
            # ================================
            required_columns = self._schema_config["required_columns"]

            train_df = train_df[required_columns]
            test_df = test_df[required_columns]

            logging.info("Required columns selected as per schema")

            # ================================
            # Split input and target
            # ================================
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # ================================
            # Drop ID columns if present
            # ================================
            drop_cols = self._schema_config.get("drop_columns", [])
            X_train.drop(columns=drop_cols, errors="ignore", inplace=True)
            X_test.drop(columns=drop_cols, errors="ignore", inplace=True)

            logging.info("Dropped ID columns if present")

            # ================================
            # Transformation
            # ================================
            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Feature transformation completed")

            # ================================
            # Handle imbalance (TRAIN ONLY)
            # ================================
            smt = SMOTEENN(sampling_strategy="minority")

            X_train_final, y_train_final = smt.fit_resample(
                X_train_transformed, y_train
            )

            logging.info("SMOTEENN applied to training data")

            # ================================
            # Concatenate features + target
            # ================================
            train_arr = np.c_[X_train_final, y_train_final.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # ================================
            # Save artifacts
            # ================================
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise MyException(e, sys) from e
