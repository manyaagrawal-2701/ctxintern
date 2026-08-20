# ==============================================================================
# SCRIPT: train.py
# PURPOSE: Orchestrates the entire Machine Learning Training Pipeline.
#          Loads raw Excel data, validates, cleans, engineers features,
#          splits into train/test sets, fits preprocessing transformations,
#          compares classification models, saves the best model, and exports
#          analytical reports and charts.
# ==============================================================================

import logging
import joblib
# train_test_split divides our dataset into:
# - Training Set (80%): Used to teach the model.
# - Test Set (20%): Kept hidden from the model, used only to evaluate its real-world performance.
from sklearn.model_selection import train_test_split

from config import Config
from core.dataloader import DataLoader
from core.model_trainer import ModelTrainer
from core.preprocessing import DataPreprocessor
from core.report_generator import ReportGenerator

# Ensure logs directory exists and configure the file-based diary logger
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=Config.LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates loading data, preprocessing, training models, and saving artifacts."""

    def __init__(self) -> None:
        """
        Constructor. Creates instances of our helper classes: Loader, Preprocessor,
        Trainer, and ReportGenerator.
        """
        self.loader = DataLoader(
            customer_path=Config.CUSTOMER_DATA_PATH,
            car_path=Config.CAR_DATA_PATH
        )
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.report_generator = ReportGenerator()

    def run(self) -> None:
        """Runs the entire training execution block."""
        print("Starting machine learning training pipeline...")
        LOGGER.info("Starting training pipeline execution.")

        # -----------------------------------------------------
        # 1. Load Datasets
        # -----------------------------------------------------
        print("Loading Excel datasets...")
        customers = self.loader.load_customer_data()
        cars = self.loader.load_car_data()

        # -----------------------------------------------------
        # 2. Find and Validate Target Label Column
        # -----------------------------------------------------
        target = self._target_column(customers)
        self.loader.validate_data(customers, required_columns=[target])

        # -----------------------------------------------------
        # 3. Merge Datasets for Analytical Reports
        # -----------------------------------------------------
        print("Merging datasets for visual reports...")
        merged_report_df = self.loader.merge_data(customers, cars)

        # Pre-clean the merged report variables to prevent Matplotlib crashes
        merged_report_df = self.preprocessor.normalize_columns(merged_report_df)
        merged_report_df = self.preprocessor.create_annual_income(merged_report_df)
        merged_report_df["TargetCarSegment"] = merged_report_df["TargetCarSegment"].map(self.preprocessor.SEGMENT_MAPPING).fillna("SUV")

        # -----------------------------------------------------
        # 4. Prepare Features & Target
        # -----------------------------------------------------
        print("Cleaning and engineering customer demographic features...")
        X, y = self.preprocessor.prepare_training_data(customers, target)

        print("\nTarget Segment Distribution:")
        print(y.value_counts())

        print(f"\nTraining model on features: {list(X.columns)}")
        print(f"Total dataset size: {X.shape[0]} rows, {X.shape[1]} features.")

        # -----------------------------------------------------
        # 5. Train-Test Split (80% Train, 20% Test)
        # -----------------------------------------------------
        # - test_size=0.20: Splits 80-20.
        # - random_state=42: A "seed" number ensuring the split is exactly repeatable.
        # - stratify=y: Ensures that both train and test splits have equal percentages
        #   of SUVs, Hatchbacks, EVs, etc., so the evaluation remains fair.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # -----------------------------------------------------
        # 6. Fit Column Transformer & Transform Data
        # -----------------------------------------------------
        print("Fitting preprocessor pipeline (scaling and encoding)...")
        # Build the router pipelines based on training variables
        self.preprocessor.build_preprocessor(X_train)
        
        # Learn scaling averages and categories ONLY from the training split, 
        # then transform both the training and test sets.
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        # -----------------------------------------------------
        # 7. Model Training & Comparison
        # -----------------------------------------------------
        print("Training classifiers (Random Forest, Gradient Boosting, SVM, XGBoost)...")
        results = self.trainer.compare_models(
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test
        )

        # Display performance logs in console
        print("\n===================================")
        print("Model Performance Matrix:")
        print("===================================\n")
        print(results.to_string(index=False))
        print("\n===================================")

        # -----------------------------------------------------
        # 8. Save Best Model and Preprocessor Assets
        # -----------------------------------------------------
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Save the winning model to models/best_model.pkl
        self.trainer.save_best_model(Config.MODEL_PATH)

        # Save the preprocessing parameters to models/preprocessor.pkl
        joblib.dump(self.preprocessor.preprocessor, Config.PREPROCESSOR_PATH)
        LOGGER.info("Saved fitted preprocessor object to: %s", Config.PREPROCESSOR_PATH)

        # -----------------------------------------------------
        # 9. Generate Excel and PDF Reports
        # -----------------------------------------------------
        print("Generating Excel and PDF reports...")
        results.to_excel(Config.REPORT_DIR / "model_comparison.xlsx", index=False)

        self.report_generator.generate_excel_report(merged_report_df, "training_data_report.xlsx")
        self.report_generator.generate_pdf_report(merged_report_df, "customer_analysis_report.pdf")

        # Create the initial visualization charts from our merged historical data
        self.report_generator.create_visualizations(merged_report_df)

        print("\n==================================================")
        print("Training Completed Successfully!")
        print(f"Selected Best Model  : {results.iloc[0]['Model']}")
        print(f"Accuracy Metric      : {results.iloc[0]['Accuracy'] * 100:.2f}%")
        print(f"F1 Score Metric      : {results.iloc[0]['F1 Score'] * 100:.2f}%")
        print(f"Saved Best Model     : {Config.MODEL_PATH}")
        print(f"Saved Preprocessor   : {Config.PREPROCESSOR_PATH}")
        print(f"PDF Analysis Report  : {Config.REPORT_DIR / 'customer_analysis_report.pdf'}")
        print("==================================================\n")

    @staticmethod
    def _target_column(df) -> str:
        """Helper to find the target column for prediction in the raw dataset."""
        possible_targets = ["TargetCarSegment", "Preferred Car Segment", "PreferredVehicleType"]
        for col in possible_targets:
            if col in df.columns:
                return col
        raise ValueError("Target class column not found in customer dataset.")


if __name__ == "__main__":
    TrainingPipeline().run()