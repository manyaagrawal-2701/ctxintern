# ==============================================================================
# CLASS: DataPreprocessor
# PURPOSE: Cleans raw columns, parses income range bounds, creates engineered 
#          features (min/max income, slab width, affordability ratio), and 
#          scales/encodes inputs via Scikit-Learn pipelines.
# ==============================================================================

import logging
import re
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Diary logger specifically for debugging data cleaning steps.
LOGGER = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data cleaning, feature engineering, scaling, encoding,
    and preparation of datasets for training and prediction.
    """

    # Dictionary mapping messy Excel column headers to standardized variable names.
    COLUMN_ALIASES = {
        "Profession": "Occupation",
        "Monthly Income Slab": "MonthlyIncome",
        "Budget (₹ Lakh)": "Budget",
        "Budget (Rs Lakh)": "Budget",
        "Budget (â‚¹ Lakh)": "Budget",
        "Preferred Car Segment": "TargetCarSegment",
        "Preferred Vehicle Type": "PreferredVehicleType",
        "Fuel Preference": "FuelPreference",
        "Purpose of Purchase": "PurchasePurpose",
        "Daily Running": "DailyRunningKM",
        "Family Size": "FamilySize",
        "Marital Status": "MaritalStatus",
    }

    # Standard feature list used for ML training and inference
    FEATURE_COLUMNS = [
        "Age",
        "Gender",
        "MaritalStatus",
        "Occupation",
        "Min_Monthly_Income",
        "Max_Monthly_Income",
        "Income_Slab_Width",
        "City",
        "FamilySize",
        "FuelPreference",
        "Budget",
        "Budget_to_Min_Income_Ratio",
        "Budget_to_Max_Income_Ratio",
        "LargeFamily",
    ]

    # Maps messy target labels in raw datasets to 6 standard categories in inventory.
    # Pickup Truck maps to SUV (lifestyle 4x4 / utility vehicle) per domain research.
    SEGMENT_MAPPING = {
        "EV": "EV",
        "SUV": "SUV",
        "Sedan": "Sedan",
        "Compact SUV": "SUV",
        "Hatchback": "Hatchback",
        "Pickup Truck": "Pickup Truck",
        "Pickup": "Pickup Truck",
        "MUV": "MUV",
        "Luxury Sedan": "Luxury",
    }

    def __init__(self) -> None:
        """Constructor. Sets the fitted preprocessor pipeline to None at startup."""
        self.preprocessor = None

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames messy Excel headers into clean standard names using COLUMN_ALIASES."""
        df = df.copy()
        return df.rename(
            columns={old: new for old, new in self.COLUMN_ALIASES.items() if old in df.columns}
        )

    def parse_income_slabs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced Income Range Extractor:
        Extracts Min_Monthly_Income, Max_Monthly_Income, Income_Slab_Width, and AnnualIncome.
        Handles both monthly ('a month') and yearly ('a year') income descriptions seamlessly.
        """
        df = df.copy()
        if "MonthlyIncome" not in df.columns:
            # Default zero initializations if missing
            df["Min_Monthly_Income"] = 0.0
            df["Max_Monthly_Income"] = 0.0
            df["Income_Slab_Width"] = 0.0
            df["AnnualIncome"] = 0.0
            return df

        def extract_bounds(val) -> Tuple[float, float]:
            if pd.isna(val):
                return (0.0, 0.0)
            if isinstance(val, (int, float)):
                v = float(val)
                return (v, v)

            s_val = str(val)
            # Find numbers ignoring commas
            numbers = [float(x.replace(",", "")) for x in re.findall(r"[\d,]+", s_val)]

            if not numbers:
                return (0.0, 0.0)

            # Check if text specifies yearly income
            is_yearly = "year" in s_val.lower() or "annual" in s_val.lower()

            if len(numbers) == 1:
                min_v = max_v = numbers[0]
            else:
                min_v, max_v = numbers[0], numbers[1]

            if is_yearly:
                min_v /= 12.0
                max_v /= 12.0

            return (min_v, max_v)

        bounds = df["MonthlyIncome"].apply(extract_bounds)
        df["Min_Monthly_Income"] = [b[0] for b in bounds]
        df["Max_Monthly_Income"] = [b[1] for b in bounds]
        df["Income_Slab_Width"] = df["Max_Monthly_Income"] - df["Min_Monthly_Income"]

        # Calculate Min and Max Annual Incomes without midpoint forcing
        df["Min_Annual_Income"] = df["Min_Monthly_Income"] * 12.0
        df["Max_Annual_Income"] = df["Max_Monthly_Income"] * 12.0
        df["AnnualIncome"] = (df["Min_Annual_Income"] + df["Max_Annual_Income"]) / 2.0

        return df

    def create_annual_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backwards compatible alias for parse_income_slabs."""
        return self.parse_income_slabs(df)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates engineered features:
        - Budget: Set lowest entry-level car budget in India to default 5.0 Lakhs (e.g. Alto / Kwid).
        - Budget_to_Min_Income_Ratio & Budget_to_Max_Income_Ratio: Direct ratio calculations using min and max bounds.
        - LargeFamily (0 or 1): True if family size is 5 or more (identifies MUV buyers).
        """
        df = df.copy()
        df = self.normalize_columns(df)
        df = self.parse_income_slabs(df)

        # 1. Budget Fallback: Default to 5.0 Lakhs (entry-level car budget in Indian market)
        if "Budget" in df.columns:
            df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce").fillna(5.0)
        else:
            df["Budget"] = 5.0

        # 2. Direct Ratios using Min and Max Monthly Income (Budget in Lakhs / Monthly Income in Thousands)
        min_monthly_k = (df["Min_Monthly_Income"] / 1000.0).replace(0, 1.0)
        max_monthly_k = (df["Max_Monthly_Income"] / 1000.0).replace(0, 1.0)

        df["Budget_to_Min_Income_Ratio"] = df["Budget"] / min_monthly_k
        df["Budget_to_Max_Income_Ratio"] = df["Budget"] / max_monthly_k
        df["IncomeToBudgetRatio"] = df["AnnualIncome"] / (df["Budget"] * 100000.0).replace(0, 1.0)

        # 3. Family Size Indicators
        if "FamilySize" in df.columns:
            df["FamilySize"] = pd.to_numeric(df["FamilySize"], errors="coerce").fillna(4)
            df["LargeFamily"] = (df["FamilySize"] >= 5).astype(int)
        else:
            df["FamilySize"] = 4
            df["LargeFamily"] = 0

        return df

    def prepare_training_data(
        self, df: pd.DataFrame, target_column: str = "TargetCarSegment"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares raw datasets for training by normalizing headers, engineering features,
        mapping targets to standard categories (defaulting fallback to 'Hatchback'),
        and splitting into Features (X) and Target Label (y).
        """
        target_column = self.COLUMN_ALIASES.get(target_column, target_column)

        df = self.normalize_columns(df)
        df = self.feature_engineering(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset: {df.columns}")

        # Map target segments to standard classes. Default fallback is 'Hatchback' (most common entry car in India)
        df[target_column] = df[target_column].map(self.SEGMENT_MAPPING).fillna("Hatchback")

        X = df[self.FEATURE_COLUMNS]
        y = df[target_column]

        return X, y

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Builds a Scikit-Learn ColumnTransformer pipeline.
        Imputation is handled cleanly inside the pipeline via SimpleImputer to prevent data leakage.
        """
        categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        numerical_cols = list(X.select_dtypes(exclude=["object", "category"]).columns)

        # 1. Pipeline for numeric features
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # 2. Pipeline for text features
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # 3. Route columns to pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
            ]
        )

        return self.preprocessor

    def fit_transform(self, X: pd.DataFrame):
        """Fits preprocessor on X and returns transformed matrix."""
        if self.preprocessor is None:
            self.build_preprocessor(X)
        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        """Transforms X using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted.")
        return self.preprocessor.transform(X)