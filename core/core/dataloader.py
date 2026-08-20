# ==============================================================================
# CLASS: DataLoader
# PURPOSE: Load customer profiles and car specifications from Excel or CSV files,
#          clean up spacer rows, validate headers, dynamically match columns,
#          and merge datasets on demand.
# ==============================================================================

import logging
from pathlib import Path
import re
from typing import Iterable
import pandas as pd

# Create a logger object specifically for this module.
# Think of this like a "diary recorder" that writes messages to logs/app.log.
# __name__ stamps this file's name (core.dataloader) on every log message.
LOGGER = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading, validating, and merging customer and car datasets
    from Excel (.xlsx, .xls) and CSV (.csv) files.
    """

    def __init__(self, customer_path: str | Path, car_path: str | Path) -> None:
        """
        Constructor method. Initializes file paths for customers and cars.
        Convert paths to 'Path' objects to handle slashes correctly on Windows and Mac.
        """
        self.customer_path = Path(customer_path)
        self.car_path = Path(car_path)

    def load_customer_data(self) -> pd.DataFrame:
        """
        Loads customer Excel or CSV file.
        Calls the helper method _read_file to handle validation and reading safely.
        """
        return self._read_file(self.customer_path, "Customer")

    def load_car_data(self) -> pd.DataFrame:
        """
        Loads car Excel or CSV file and removes blank visual spacer rows.
        """
        df = self._read_file(self.car_path, "Car")
        
        # Spacer rows (empty rows inside Excel files used for layouts) will crash
        # math calculations. We drop rows where 'Brand' or 'Model' name is missing.
        if "Brand" in df.columns:
            df = df.dropna(subset=["Brand"])
        elif "Model" in df.columns:
            df = df.dropna(subset=["Model"])
        return df

    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """
        Checks if the table is loaded correctly and contains expected columns.
        """
        # 1. Sanity Check: If the spreadsheet has 0 rows, raise a clean error.
        if df.empty:
            raise ValueError("Dataset is empty.")

        # 2. Column Check: Ensure all required headers are present.
        if required_columns:
            missing = [
                col for col in required_columns
                if col not in df.columns
            ]
            if missing:
                raise ValueError(
                    f"Missing required columns: {', '.join(missing)}"
                )

        # 3. Missing Cell Check: Log a warning if there are empty cells in the table.
        if df.isnull().sum().sum() > 0:
            LOGGER.warning("Dataset contains missing values.")

        LOGGER.info("Dataset validation completed.")
        return df

    def merge_data(
        self,
        customer_df: pd.DataFrame,
        car_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges customer table with car table based on segment preference.
        This denormalizes data for analytics and model training.
        """
        # 1. Dynamically find which column represents car segment/type in customer table.
        segment_column = self._find_segment_column(customer_df)

        # 2. Check: If customer segment preference column is missing, skip the merge.
        if segment_column is None:
            LOGGER.warning("No matching customer segment column found. Merge skipped.")
            return customer_df.copy()

        # 3. Check: If the car table doesn't have a 'Segment' or vehicle type column, skip the merge.
        car_segment_column = self._find_segment_column(car_df) or ("Segment" if "Segment" in car_df.columns else None)
        if car_segment_column is None:
            LOGGER.warning("Car dataset does not contain 'Segment' or vehicle type column.")
            return customer_df.copy()

        # 4. Perform a 'Left Join' (keep all customers, even if their preference is not in inventory).
        # Suffixes appends '_car' to duplicate columns (e.g. 'SafetyRating' -> 'SafetyRating_car').
        merged = customer_df.merge(
            car_df,
            left_on=segment_column,
            right_on=car_segment_column,
            how="left",
            suffixes=("", "_car"),
        )

        LOGGER.info(
            "Datasets merged successfully using columns '%s' and '%s'. Shape: %s",
            segment_column,
            car_segment_column,
            merged.shape,
        )
        return merged

    @staticmethod
    def _read_file(path: Path, dataset_name: str) -> pd.DataFrame:
        """
        A static helper method to load Excel (.xlsx, .xls) or CSV (.csv) files safely.
        """
        # 1. Check if the file exists at the address.
        if not path.exists():
            raise FileNotFoundError(f"{dataset_name} dataset not found:\n{path}")

        ext = path.suffix.lower()
        # 2. Check if the file is a supported spreadsheet format.
        if ext not in {".xlsx", ".xls", ".csv"}:
            raise ValueError(
                f"{dataset_name} file must be an Excel (.xlsx, .xls) or CSV (.csv) file."
            )

        # 3. Read the file safely inside a try-except box.
        try:
            if ext == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            LOGGER.info(
                "%s dataset loaded successfully from %s. Shape: %s",
                dataset_name,
                ext,
                df.shape,
            )
            return df
        except Exception as e:
            LOGGER.exception("Failed to load %s dataset.", dataset_name)
            raise e

    @staticmethod
    def _read_excel(path: Path, dataset_name: str) -> pd.DataFrame:
        """
        Backwards-compatible alias for loading Excel or CSV datasets.
        """
        return DataLoader._read_file(path, dataset_name)

    @staticmethod
    def _find_segment_column(df: pd.DataFrame) -> str | None:
        """
        Smartly searches table columns for car segment / vehicle type preferences.
        Handles exact matches first, then falls back to keyword/pattern matching.
        """
        # 1. Primary candidates (exact matches)
        candidates = [
            "TargetCarSegment",
            "PreferredVehicleType",
            "Preferred Car Segment",
            "Segment",
            "Car_Type",
            "CarType",
            "Vehicle_Type",
            "VehicleType",
            "Car_Segment",
            "CarSegment",
        ]
        for column in candidates:
            if column in df.columns:
                return column

        # 2. Fallback: Flexible keyword regex matching
        keywords = [r"segment", r"car_?type", r"vehicle_?type", r"car_?pref", r"body_?style"]
        for col in df.columns:
            clean_col = str(col).lower().strip().replace(" ", "_")
            for kw in keywords:
                if re.search(kw, clean_col):
                    return col

        return None

    @staticmethod
    def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """
        Static helper method. Searches the table columns and returns the first column 
        name that matches any name in our candidate list.
        """
        for column in candidates:
            if column in df.columns:
                return column
        return None