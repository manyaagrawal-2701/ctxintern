import os
from pathlib import Path


class Config:
    """Central application configuration."""

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    REPORT_DIR = BASE_DIR / "reports"
    LOG_DIR = BASE_DIR / "logs"

    CUSTOMER_DATA_PATH = DATA_DIR / "New_Customer_Data_1500 EXCEL.xlsx"
    CAR_DATA_PATH = DATA_DIR / "Car Dealer list.xlsx"
    MODEL_PATH = MODEL_DIR / "best_model.pkl"
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
    LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "manya@123")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "car_recommendation")
