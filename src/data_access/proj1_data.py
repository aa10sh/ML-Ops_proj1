import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException


class Proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Optimized export of MongoDB collection as DataFrame
        (low-latency, schema-aligned)
        """
        try:
            # Get collection
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            print("Fetching data from mongoDB (optimized)")

            # ================================
            # ONLY required columns
            # ================================
            required_columns = [
                "subscription_length",
                "vehicle_age",
                "customer_age",
                "region_density",
                "region_code",
                "segment",
                "model",
                "gross_weight",
                "displacement",
                "length",
                "claim_status",
            ]

            projection = {col: 1 for col in required_columns}
            projection["_id"] = 0

            # ================================
            # Batched cursor
            # ================================
            cursor = (
                collection
                .find({}, projection)
                .batch_size(5000)
            )

            # ================================
            # Fast DataFrame creation
            # ================================
            df = pd.DataFrame.from_records(cursor)

            print(f"Data fetched with len: {len(df)}")

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)
