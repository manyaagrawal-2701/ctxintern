# ==============================================================================
# CLASS: DatabaseManager
# PURPOSE: Manages connection to local MySQL server. Creates the database and
#          log table dynamically, handles logging predictions, and retrieves
#          aggregated stats for the dashboard.
# ==============================================================================

import logging
from typing import Any
import mysql.connector
import pandas as pd
from mysql.connector import Error
from config import Config

# Logger to log SQL database connection or query issues.
LOGGER = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(
        self,
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DATABASE,
        port=Config.MYSQL_PORT,
    ):
        """
        Constructor. Initializes MySQL configurations and automatically creates 
        the database and tables if they don't exist yet on the server.
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

        # Automatically build the schema
        self.create_database()
        self.create_table()

    # --------------------------------------------------------------------------
    # CONNECTIONS & SCHEMA SETUP
    # --------------------------------------------------------------------------

    def get_connection(self, include_database=True):
        """
        Creates and returns a new active connection to the MySQL server.
        - include_database=False is used when creating the database itself.
        """
        config = {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "port": self.port,
        }

        if include_database:
            config["database"] = self.database

        # Open the connection gate
        return mysql.connector.connect(**config)

    def create_database(self):
        """
        Connects to the server and creates the target database if missing.
        """
        try:
            # We connect without targeting a specific database (include_database=False)
            # because the database doesn't exist yet.
            with self.get_connection(False) as conn:
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                conn.commit()
        except Error as e:
            LOGGER.error(e)

    def create_table(self):
        """
        Creates the 'users' and 'predictions' tables inside the database if missing.
        Handles alter tables gracefully for schema evolution.
        """
        users_query = """
        CREATE TABLE IF NOT EXISTS users(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        predictions_query = """
        CREATE TABLE IF NOT EXISTS predictions(
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id INT,
            age INT,
            gender VARCHAR(30),
            city VARCHAR(100),
            budget FLOAT,
            fuel_preference VARCHAR(50),
            transmission VARCHAR(50),
            vehicle_type VARCHAR(50),
            predicted_segment VARCHAR(100),
            recommended_brand VARCHAR(100),
            recommended_model VARCHAR(100),
            price FLOAT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(users_query)
                cursor.execute(predictions_query)

                # Migration Check: does user_id column exist in predictions?
                cursor.execute("SHOW COLUMNS FROM predictions LIKE 'user_id'")
                if not cursor.fetchone():
                    cursor.execute(
                        "ALTER TABLE predictions ADD COLUMN user_id INT, "
                        "ADD FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"
                    )
                conn.commit()
        except Error as e:
            LOGGER.error(e)

    # --------------------------------------------------------------------------
    # USER AUTHENTICATION QUERIES
    # --------------------------------------------------------------------------

    def create_user(self, username: str, password_hash: str) -> int | None:
        """
        Registers a new user inside the database.
        Returns the new user's ID, or None if username is taken.
        """
        query = "INSERT INTO users (username, password_hash) VALUES (%s, %s)"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username.strip(), password_hash))
                conn.commit()
                return cursor.lastrowid
        except Error as e:
            LOGGER.error("Failed to create user: %s", e)
            return None

    def get_user_by_username(self, username: str) -> dict | None:
        """
        Retrieves a user record by username for verification check.
        """
        query = "SELECT id, username, password_hash FROM users WHERE username = %s"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username.strip(),))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "username": row[1],
                        "password_hash": row[2]
                    }
        except Error as e:
            LOGGER.error("Failed to get user: %s", e)
        return None

    # --------------------------------------------------------------------------
    # WRITING & READING LOG ENTRIES (USER SPECIFIC)
    # --------------------------------------------------------------------------

    def save_prediction(self, prediction: dict, user_id: int | None = None):
        """
        Inserts a completed customer match transaction record into the table.
        - Uses parameterized query placeholders (%s) to prevent SQL injection bugs.
        """
        query = """
        INSERT INTO predictions(
            user_id,
            age,
            gender,
            city,
            budget,
            fuel_preference,
            transmission,
            vehicle_type,
            predicted_segment,
            recommended_brand,
            recommended_model,
            price
        )
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            user_id,
            prediction["age"],
            prediction["gender"],
            prediction["city"],
            prediction["budget"],
            prediction["fuel_preference"],
            prediction["transmission"],
            prediction["vehicle_type"],
            prediction["predicted_segment"],
            prediction["recommended_brand"],
            prediction["recommended_model"],
            prediction["price"]
        )

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()  # commit saves the row permanently
        except Error as e:
            LOGGER.error(e)

    def get_prediction_history(self, user_id: int | None = None):
        """
        Fetches predictions history log filtered specifically for the logged-in user.
        Returns a Pandas DataFrame sorted with the newest entries at row 0.
        """
        try:
            with self.get_connection() as conn:
                if user_id is not None:
                    query = "SELECT * FROM predictions WHERE user_id = %s ORDER BY id DESC"
                    params = (user_id,)
                else:
                    query = "SELECT * FROM predictions ORDER BY id DESC"
                    params = ()

                # Note: pandas read_sql warning can be bypassed using simple cursor fetch if needed,
                # but read_sql remains simple and powerful.
                return pd.read_sql(query, conn, params=params)
        except Exception as e:
            LOGGER.error(e)
            return pd.DataFrame()

    def get_dashboard_metrics(self, user_id: int | None = None):
        """
        Aggregates logs from the history table to calculate dashboard KPI metrics.
        - total_predictions: Total records in the table.
        - most_recommended_brand: The brand recommended most frequently.
        - average_budget: Median of customer budgets.
        """
        history = self.get_prediction_history(user_id)

        # If table is empty, return clean default statistics
        if history.empty:
            return {
                "total_predictions": 0,
                "most_recommended_brand": "-",
                "average_budget": 0,
                "average_price": 0,
                "segment_distribution": {},
                "brand_distribution": {}
            }

        # Calculate means and modes from logs
        return {
            "total_predictions": len(history),
            "most_recommended_brand": history["recommended_brand"].mode()[0],
            "average_budget": round(history["budget"].mean(), 2),
            "average_price": round(history["price"].mean(), 2),
            "segment_distribution": history["predicted_segment"].value_counts().to_dict(),
            "brand_distribution": history["recommended_brand"].value_counts().to_dict()
        }

    def delete_history(self, user_id: int | None = None):
        """
        Deletes all predictions rows inside table associated with the current user.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if user_id is not None:
                    cursor.execute("DELETE FROM predictions WHERE user_id = %s", (user_id,))
                else:
                    cursor.execute("DELETE FROM predictions")
                conn.commit()
        except Exception as e:
            LOGGER.error(e)

    def close(self):
        """Placeholder for connection teardowns."""
        pass