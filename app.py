# ==============================================================================
# FLASK WEB HOST: app.py
# PURPOSE: Coordinates the DriveWise web application. Handles landing page routing,
#          user login/signup authentication, prediction calculation, dashboard metrics,
#          and history logging with strict user isolation.
# ==============================================================================

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from core.database_manager import DatabaseManager
from core.recommendation_engine import CarRecommendationEngine
from core.report_generator import ReportGenerator
from core.preprocessing import DataPreprocessor

# Configure logging path and logging string formats.
logging.basicConfig(
    filename=Config.LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


class CarRecommendationApp:
    """Class-based Flask application for the DriveWise Car Recommendation System."""

    # Simple rules to fall back on if the ML model is corrupted/missing.
    SEGMENT_RULES = [
        (45, "Luxury"),
        (18, "SUV"),
        (12, "Sedan"),
        (8, "Hatchback"),
    ]

    def __init__(self) -> None:
        """
        Constructor. Initializes Flask configurations, deserializes models,
        loads inventories, and builds DB managers.
        """
        self.app = Flask(__name__)
        self.app.secret_key = Config.SECRET_KEY
        self.app.config.from_object(Config)

        # 1. Load ML trained files
        self.model = self._load_joblib(Config.MODEL_PATH)
        self.preprocessor = self._load_joblib(Config.PREPROCESSOR_PATH)
        
        # 2. Load the dealer inventory catalog
        self.car_data = self._load_car_data(Config.CAR_DATA_PATH)
        
        # 3. Instantiate helper handlers
        self.engine = CarRecommendationEngine(self.model, self.car_data)
        self.db = DatabaseManager()
        self.report_generator = ReportGenerator()
        self.data_preprocessor = DataPreprocessor()

        self._register_routes()

    def _register_routes(self) -> None:
        """Configures URL paths mapping to specific view methods."""
        # Public Views
        self.app.add_url_rule("/", view_func=self.home)
        self.app.add_url_rule("/register", view_func=self.register, methods=["GET", "POST"])
        self.app.add_url_rule("/login", view_func=self.login, methods=["GET", "POST"])
        self.app.add_url_rule("/logout", view_func=self.logout)
        
        # Guarded Views (Require Login)
        self.app.add_url_rule("/predict_form", view_func=self.predict_form)
        self.app.add_url_rule("/predict", view_func=self.predict, methods=["POST"])
        self.app.add_url_rule("/search", view_func=self.search, methods=["GET", "POST"])
        self.app.add_url_rule("/dashboard", view_func=self.dashboard)
        self.app.add_url_rule("/history", view_func=self.history)
        self.app.add_url_rule("/clear_history", view_func=self.clear_history, methods=["POST"])

    # --------------------------------------------------------------------------
    # PUBLIC VIEW ROUTE HANDLERS
    # --------------------------------------------------------------------------

    def home(self):
        """GET '/': Displays the main cover/landing page for DriveWise."""
        top_cars = self.engine.get_top_5_cars().to_dict(orient="records")
        return render_template("index.html", top_cars=top_cars)

    def register(self):
        """GET/POST '/register': Allows users to create a clean account."""
        if "user_id" in session:
            return redirect(url_for("home"))
            
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            
            if not username or not password:
                flash("Username and password are required.", "danger")
                return render_template("register.html")
                
            # Check if username exists
            existing_user = self.db.get_user_by_username(username)
            if existing_user:
                flash("Username is already taken. Try another.", "danger")
                return render_template("register.html")
                
            # Create user with secure hash
            pw_hash = generate_password_hash(password)
            user_id = self.db.create_user(username, pw_hash)
            if user_id:
                flash("Registration successful! Please login below.", "success")
                return redirect(url_for("login"))
            else:
                flash("Failed to register. Please try again.", "danger")
                
        return render_template("register.html")

    def login(self):
        """GET/POST '/login': Authenticates users and registers session."""
        if "user_id" in session:
            return redirect(url_for("home"))
            
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            
            user = self.db.get_user_by_username(username)
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                flash(f"Welcome back, {user['username']}!", "success")
                return redirect(url_for("home"))
            else:
                flash("Invalid username or password.", "danger")
                
        return render_template("login.html")

    def logout(self):
        """GET '/logout': Clears active session credentials."""
        session.clear()
        flash("You have logged out successfully.", "success")
        return redirect(url_for("home"))

    # --------------------------------------------------------------------------
    # GUARDED VIEW ROUTE HANDLERS
    # --------------------------------------------------------------------------

    def predict_form(self):
        """GET '/predict_form': Renders recommendation questionnaire."""
        if "user_id" not in session:
            flash("Please login to access the Recommendation Questionnaire.", "warning")
            return redirect(url_for("login"))
        return render_template("predict_form.html")

    def predict(self):
        """
        POST '/predict': Processes submitted form parameters, calculates matching
        scores across inventory, inserts prediction records in DB, and renders result cards.
        """
        if "user_id" not in session:
            flash("Please login to request car recommendations.", "warning")
            return redirect(url_for("login"))

        user_id = session["user_id"]

        try:
            # Gather profile dictionary
            customer = self._form_to_customer(request.form)
            
            # Compute engineered fields dynamically
            frame = pd.DataFrame([customer])
            frame = self.data_preprocessor.feature_engineering(frame)
            engineered_customer = frame.iloc[0].to_dict()
            
            customer["AnnualIncome"] = engineered_customer["AnnualIncome"]
            customer["Min_Monthly_Income"] = engineered_customer["Min_Monthly_Income"]
            customer["Max_Monthly_Income"] = engineered_customer["Max_Monthly_Income"]
            
            # Find matching segment via ML predictions or logical rules
            segment = self._predict_segment(customer)
            
            # Score and filter car models
            recommendations = self.engine.recommend_car(
                segment=segment,
                budget=customer["Budget"],
                fuel_preference=customer["FuelPreference"],
                transmission=customer["Transmission"],
                family_size=customer["FamilySize"],
                driving_experience=customer["DrivingExperience"],
                daily_running=customer["DailyRunningKM"],
                preferred_type=customer["PreferredVehicleType"]
            )
            
            if recommendations.empty:
                flash("No cars found matching your criteria. Try adjusting budget or preferences.", "warning")
                return redirect(url_for("predict_form"))
                
            top_car = recommendations.iloc[0].to_dict()
            
            # Insert log transaction record associated with logged-in user
            self.db.save_prediction(
                {
                    "age": customer["Age"],
                    "gender": customer["Gender"],
                    "city": customer["City"],
                    "budget": customer["Budget"],
                    "fuel_preference": customer["FuelPreference"],
                    "transmission": customer["Transmission"],
                    "vehicle_type": customer["PreferredVehicleType"],
                    "predicted_segment": segment,
                    "recommended_brand": top_car["Brand"],
                    "recommended_model": top_car["Model"],
                    "price": top_car["Price"],
                },
                user_id=user_id
            )

            # Regenerate user-specific analytics visualizations dynamically
            history = self.db.get_prediction_history(user_id)
            if not history.empty:
                self.report_generator.create_visualizations(history, user_id=user_id)

            return render_template(
                "result.html",
                segment=segment,
                cars=recommendations.to_dict(orient="records"),
                customer=customer,
            )
        except Exception as exc:
            LOGGER.exception("Prediction route failed.")
            flash(f"Error making recommendation: {exc}", "danger")
            return redirect(url_for("predict_form"))

    def search(self):
        """GET/POST '/search': Allows simple keywords lookup for cars."""
        if "user_id" not in session:
            flash("Please login to search car inventories.", "warning")
            return redirect(url_for("login"))

        query = request.args.get("query", "").strip()
        search_type = request.args.get("type", "brand").strip()
        results = pd.DataFrame()

        if query:
            try:
                if search_type == "brand":
                    results = self.engine.search_by_brand(query)
                elif search_type == "segment":
                    results = self.engine.search_by_segment(query)
                elif search_type == "budget":
                    try:
                        budget_val = float(query)
                        results = self.engine.search_by_budget(budget_val)
                    except ValueError:
                        flash("Please enter a numeric value for budget search.", "danger")
            except Exception as e:
                LOGGER.exception("Search execution failed.")
                flash(f"Search failed: {e}", "danger")

        cars_list = results.to_dict(orient="records") if not results.empty else []
        return render_template(
            "predict_form.html",
            search_results=cars_list,
            query=query,
            type=search_type,
            top_cars=self.engine.get_top_5_cars().to_dict(orient="records")
        )

    def clear_history(self):
        """POST '/clear_history': Deletes prediction entries associated with this user."""
        if "user_id" not in session:
            return redirect(url_for("login"))

        user_id = session["user_id"]
        try:
            self.db.delete_history(user_id)
            flash("Your prediction history has been cleared.", "success")
        except Exception as e:
            LOGGER.exception("Clear history failed.")
            flash(f"Failed to clear history: {e}", "danger")
        return redirect(url_for("history"))

    def dashboard(self):
        """GET '/dashboard': Renders dynamic user-specific dashboard metrics."""
        if "user_id" not in session:
            flash("Please login to view your metrics dashboard.", "warning")
            return redirect(url_for("login"))

        user_id = session["user_id"]
        history = self.db.get_prediction_history(user_id)
        metrics = self.db.get_dashboard_metrics(user_id)
        total_customers = self._customer_count()
        
        # Keep visualizations updated for this specific user
        if not history.empty:
            self.report_generator.create_visualizations(history, user_id=user_id)

        return render_template(
            "dashboard.html",
            metrics=metrics,
            total_customers=total_customers,
            recent=history.head(8).to_dict(orient="records") if not history.empty else [],
        )

    def history(self):
        """GET '/history': Lists user-specific query history logs."""
        if "user_id" not in session:
            flash("Please login to view your history logs.", "warning")
            return redirect(url_for("login"))

        user_id = session["user_id"]
        history = self.db.get_prediction_history(user_id)
        return render_template(
            "history.html",
            rows=history.to_dict(orient="records") if not history.empty else [],
        )

    # --------------------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------------------

    def _predict_segment(self, customer: dict[str, Any]) -> str:
        """
        Predicts suitable car segments.
        1. Scales form values and inputs them into the model.
        2. Applies consistency override check: if model outputs 'EV' but fuel pref
           is Petrol/Diesel, redirect segment logically based on budget.
        3. If ML files are missing, fallbacks to rule-based logic.
        """
        predicted = None
        if self.model is not None and self.preprocessor is not None:
            try:
                frame = pd.DataFrame([customer])
                frame = self.data_preprocessor.feature_engineering(frame)
                X_pred = frame[self.data_preprocessor.FEATURE_COLUMNS]
                
                transformed = self.preprocessor.transform(X_pred)
                predicted = self.engine.predict_customer_segment(transformed)
            except Exception:
                LOGGER.exception("ML prediction failed; executing rule-based engine.")

        fuel = str(customer.get("FuelPreference", "")).lower()

        # Enforce consistency check
        if fuel == "electric":
            predicted = "EV"
        elif predicted == "EV" and fuel in ["petrol", "diesel", "cng", "hybrid"]:
            budget = float(customer["Budget"])
            if budget >= 25.0:
                predicted = "Luxury"
            elif budget >= 15.0:
                predicted = "SUV"
            elif budget >= 10.0:
                predicted = "Sedan"
            else:
                predicted = "Hatchback"

        # Fallback Heuristics
        if not predicted:
            budget = float(customer["Budget"])
            if fuel == "electric":
                return "EV"
            if int(customer["FamilySize"]) >= 5:
                return "MUV"
            for threshold, segment in self.SEGMENT_RULES:
                if budget >= threshold:
                    return segment
            return "Hatchback"

        return predicted

    def _form_to_customer(self, form) -> dict[str, Any]:
        """Validates and parses web form parameters into a standardized schema."""
        return {
            "Age": int(form.get("Age", 30)),
            "Gender": form.get("Gender", "Male"),
            "MaritalStatus": form.get("MaritalStatus", "Single"),
            "Occupation": form.get("Occupation", "Professional"),
            "MonthlyIncome": str(form.get("MonthlyIncome", "60000")).strip(),
            "AnnualIncome": 0.0,
            "City": form.get("City", "Unknown"),
            "FamilySize": int(form.get("FamilySize", 4)),
            "DrivingExperience": float(form.get("DrivingExperience", 5.0)),
            "FuelPreference": form.get("FuelPreference", "Petrol"),
            "Budget": float(form.get("Budget", 10.0)),
            "PurchasePurpose": form.get("PurchasePurpose", "Personal"),
            "DailyRunningKM": float(form.get("DailyRunningKM", 30.0)),
            "Transmission": form.get("Transmission", "Manual"),
            "PreferredVehicleType": form.get("PreferredVehicleType", "SUV"),
        }

    def _customer_count(self) -> int:
        """Determines total customer row counts inside customer Excel sheet."""
        if Config.CUSTOMER_DATA_PATH.exists():
            try:
                return len(pd.read_excel(Config.CUSTOMER_DATA_PATH))
            except Exception:
                LOGGER.exception("Unable to read customer count from excel.")
        return 0

    @staticmethod
    def _load_joblib(path: Path):
        """Safely loads Joblib serialized models."""
        try:
            return joblib.load(path) if path.exists() else None
        except Exception:
            LOGGER.exception("Unable to load artifact: %s", path)
            return None

    @staticmethod
    def _load_car_data(path: Path) -> pd.DataFrame:
        """Loads car Excel dataset. Falls back to static list if Excel is missing."""
        if path.exists():
            try:
                df = pd.read_excel(path)
                return df
            except Exception:
                LOGGER.exception("Unable to load car Excel list.")
        
        # Hardcoded fallback list in case file is deleted
        return pd.DataFrame(
            [
                ["Maruti Suzuki", "Swift", "Hatchback", "Petrol", "Manual", 650000.0, 22.0, 5, 2.0, 2],
                ["Tata", "Nexon", "SUV", "Petrol", "Manual", 1200000.0, 17.0, 5, 5.0, 6],
                ["Honda", "City", "Sedan", "Petrol", "Manual", 1200000.0, 18.0, 5, 4.0, 6],
                ["Toyota", "Innova Hycross", "MUV", "Hybrid", "Automatic", 2000000.0, 21.0, 7, 5.0, 6],
                ["MG", "Comet EV", "EV", "Electric", "Automatic", 700000.0, 23.0, 4, 3.0, 2],
                ["BMW", "3 Series", "Luxury", "Petrol", "Automatic", 4500000.0, 15.0, 5, 5.0, 6],
            ],
            columns=["Brand", "Model", "Segment", "FuelType", "Transmission", "Price", "Mileage", "SeatingCapacity", "SafetyRating", "Airbags"]
        )


car_app = CarRecommendationApp()
app = car_app.app

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
