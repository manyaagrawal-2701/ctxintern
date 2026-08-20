# ==============================================================================
# CLASS: CarRecommendationEngine
# PURPOSE: Evaluates and scores every car in the catalog against user-submitted
#          demographics and preferences. Selects the top 5 matches.
# ==============================================================================

import logging
from typing import Any, List
import pandas as pd

# Logger specifically for tracking car matchmaking operations.
LOGGER = logging.getLogger(__name__)


class CarRecommendationEngine:
    """
    Predicts segment and recommends the best matching cars using demographic
    classification and detailed customer preference scoring.
    """

    # Maps different spelling variations of car segments in dealer list (like "compact suv" 
    # or "mpv") to our standardized 6 main categories (like "SUV" and "MUV").
    CAR_SEGMENT_MAP = {
        "hatchback": "Hatchback",
        "premium hatchback": "Hatchback",
        "sedan": "Sedan",
        "suv": "SUV",
        "compact suv": "SUV",
        "micro suv": "SUV",
        "mpv": "MUV",
        "muv": "MUV",
        "pickup truck": "Pickup Truck",
        "pickup": "Pickup Truck",
        "ev hatchback": "EV",
        "ev": "EV",
    }

    def __init__(self, model: Any | None, car_data: pd.DataFrame) -> None:
        """
        Constructor. Takes the trained ML model object and the raw car list,
        then normalizes columns (e.g. converting price from Rupees to Lakhs).
        """
        self.model = model
        self.car_data = self._normalize_car_data(car_data)

    def predict_customer_segment(self, features) -> str:
        """
        Takes scaled numerical demographic features (from the web form) 
        and uses the ML model to predict the most suitable segment category.
        """
        if self.model is None:
            raise ValueError("Prediction model is not loaded.")
        prediction = self.model.predict(features)
        return str(prediction[0])

    def recommend_car(
        self,
        segment: str,
        budget: float,
        fuel_preference: str | None = None,
        transmission: str | None = None,
        family_size: int = 4,
        driving_experience: float = 5.0,
        daily_running: float = 30.0,
        preferred_type: str | None = None,
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Scores all cars in the catalog using demographic forecasts, explicit user preferences,
        and budget bounds, then returns the top 'limit' (default 5) matches.
        """
        budget = float(budget)
        cars = self.car_data.copy()

        # 1. Score every single car in our database without pre-filtering.
        cars["Score"] = cars.apply(
            lambda row: self.calculate_score(
                row=row,
                customer_budget=budget,
                predicted_segment=segment,
                fuel_preference=fuel_preference,
                transmission=transmission,
                family_size=family_size,
                driving_experience=driving_experience,
                daily_running=daily_running,
                preferred_type=preferred_type,
            ),
            axis=1,
        )

        # 2. Sort recommendations so the highest scores sit at the top.
        ranked = self.rank_recommendations(cars)

        # 3. Generate natural language explanations for the matches.
        ranked = ranked.copy()
        ranked["Reason"] = ranked.apply(
            lambda row: self.generate_reason(
                row=row,
                budget=budget,
                predicted_segment=segment,
                fuel_preference=fuel_preference,
                transmission=transmission,
                family_size=family_size,
                driving_experience=driving_experience,
                daily_running=daily_running,
                preferred_type=preferred_type,
            ),
            axis=1,
        )

        # 4. Enforce Segment Diversity (cap at most 2 cars from the same segment in recommendations)
        diverse_selections = []
        segment_counts = {}
        
        for idx, row in ranked.iterrows():
            car_seg = row["Segment"]
            count = segment_counts.get(car_seg, 0)
            if count < 2:  # Limit same-segment results to at most 2
                diverse_selections.append(row)
                segment_counts[car_seg] = count + 1
            if len(diverse_selections) >= limit:
                break
                
        # If we couldn't fill the limit due to constraints, fall back to simple ranked list
        if len(diverse_selections) < limit:
            return ranked.head(limit)
            
        return pd.DataFrame(diverse_selections)

    def rank_recommendations(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """Ranks recommendations in descending order of score."""
        return recommendations_df.sort_values(by="Score", ascending=False)

    def calculate_score(
        self,
        row: pd.Series,
        customer_budget: float,
        predicted_segment: str,
        fuel_preference: str | None,
        transmission: str | None,
        family_size: int,
        driving_experience: float,
        daily_running: float,
        preferred_type: str | None = None,
    ) -> float:
        """
        Calculates a recommendation score from 0 to 100 for a single car row.
        Think of this as a weight matrix where different matches add points:
        """
        score = 0.0

        # 1. Segment Match & Preference Boost (up to 25 points)
        # If the car matches the ML predicted segment, add +10 points.
        # If it matches the user's explicitly preferred vehicle type, add +15 points.
        car_segment = str(row["Segment"]).lower()
        if predicted_segment and car_segment == predicted_segment.lower():
            score += 10.0
        if preferred_type and car_segment == preferred_type.lower():
            score += 15.0

        # 2. Budget Proximity (up to 30 points)
        price = float(row["Price"])
        if price <= customer_budget:
            # Under budget: closer to budget limit gets more points
            diff = customer_budget - price
            score += max(0.0, 30.0 - (diff * 1.0))
        else:
            # Over budget: heavily penalized to keep recommendations affordable
            excess = price - customer_budget
            score += max(0.0, 30.0 - (excess * 15.0))

        # 3. Transmission Match (up to 15 points)
        if transmission and str(row["Transmission"]).lower() == transmission.lower():
            score += 15.0

        # 4. Fuel Preference Match (up to 15 points)
        # Supports EV equivalents (like Electric matches EV)
        if fuel_preference:
            if fuel_preference.lower() == "electric" and str(row["FuelType"]).lower() in ["electric", "ev"]:
                score += 15.0
            elif str(row["FuelType"]).lower() == fuel_preference.lower():
                score += 15.0

        # 5. Family Seating Capacity Fit (up to 10 points)
        seating = int(row["SeatingCapacity"])
        if family_size >= 5:
            # Large family needs 6 or 7 seater
            if seating >= 6:
                score += 10.0
            else:
                score += 1.0  # Penalize small cars for large families
        else:
            # Small family prefers 5-seater or less
            if seating <= 5:
                score += 10.0
            else:
                score += 7.0  # Big car is fine, but not strictly necessary

        # 6. Driving Experience & Safety Boost (up to 10 points)
        safety_rating = float(row["SafetyRating"])
        airbags = int(row["Airbags"])
        if driving_experience <= 2.0:
            # New Driver: Weight safety heavily (NCAP stars and airbag count)
            score += (safety_rating * 1.2) + (airbags * 0.6)
        else:
            # Experienced Driver: standard weighting
            score += (safety_rating * 0.8) + (airbags * 0.4)

        # 7. Daily Running Economy Boost (up to 5 points)
        mileage = float(row["Mileage"])
        if daily_running >= 50.0:
            # High daily usage: prioritize electric/hybrid or high mileage cars
            if str(row["FuelType"]).lower() in ["electric", "ev", "hybrid"]:
                score += 5.0
            else:
                score += min(5.0, mileage * 0.2)
        else:
            score += min(3.0, mileage * 0.1)

        return round(score, 2)

    def generate_reason(
        self,
        row: pd.Series,
        budget: float,
        predicted_segment: str,
        fuel_preference: str | None,
        transmission: str | None,
        family_size: int,
        driving_experience: float,
        daily_running: float,
        preferred_type: str | None = None,
    ) -> str:
        """
        Generates a user-friendly sentence explaining why this car is recommended.
        Joins sentences together using pipes (e.g. "Fits budget | High safety...").
        """
        reasons = []
        price = float(row["Price"])
        car_segment = str(row["Segment"]).lower()

        # Segment match reason
        if preferred_type and car_segment == preferred_type.lower():
            reasons.append(f"Matches your preferred {row['Segment']} type")
        elif predicted_segment and car_segment == predicted_segment.lower():
            reasons.append(f"Matches predicted {row['Segment']} demographic category")

        # Budget reason
        if price <= budget:
            reasons.append(f"Fits your ₹{budget:.1f} Lakh budget (Priced at ₹{price:.1f} Lakh)")
        else:
            reasons.append(f"Priced at ₹{price:.1f} Lakh (slightly exceeds ₹{budget:.1f} Lakh budget)")

        # Transmission reason
        if transmission and str(row["Transmission"]).lower() == transmission.lower():
            reasons.append(f"Matches your {transmission} preference")

        # Fuel reason
        if fuel_preference and (
            str(row["FuelType"]).lower() == fuel_preference.lower()
            or (fuel_preference.lower() == "electric" and str(row["FuelType"]).lower() in ["electric", "ev"])
        ):
            reasons.append(f"{row['FuelType']} fuel type")

        # Seating capacity reason
        seating = int(row["SeatingCapacity"])
        if family_size >= 5 and seating >= 6:
            reasons.append(f"{seating} seats for family of {family_size}")
        elif family_size < 5 and seating <= 5:
            reasons.append(f"Comfortable {seating}-seater")

        # Safety & experience reason
        safety = float(row["SafetyRating"])
        airbags = int(row["Airbags"])
        if driving_experience <= 2.0:
            reasons.append(f"Extra safety ({safety}★ rating, {airbags} airbags) for new driver")
        else:
            reasons.append(f"{safety}★ NCAP safety & {airbags} airbags")

        # Daily usage reason
        mileage = float(row["Mileage"])
        if daily_running >= 50.0:
            if str(row["FuelType"]).lower() in ["electric", "ev"]:
                reasons.append(f"Electric model is ideal for {daily_running} km/day running")
            elif str(row["FuelType"]).lower() == "hybrid":
                reasons.append(f"Hybrid ({mileage} km/l) is cost-effective for {daily_running} km/day")
            else:
                reasons.append(f"Highly economical ({mileage} km/l) for {daily_running} km/day")

        return " | ".join(reasons)

    def search_by_budget(self, budget: float) -> pd.DataFrame:
        """Search cars costing up to the specified budget."""
        return self.car_data[self.car_data["Price"] <= budget].sort_values(by="Price")

    def search_by_brand(self, brand: str) -> pd.DataFrame:
        """Search cars by brand name (case-insensitive substring search)."""
        return self.car_data[self.car_data["Brand"].str.contains(brand, case=False, na=False)]

    def search_by_segment(self, segment: str) -> pd.DataFrame:
        """Search cars by normalized segment name."""
        return self.car_data[self.car_data["Segment"].str.contains(segment, case=False, na=False)]

    def get_top_5_cars(self) -> pd.DataFrame:
        """Returns the top 5 cars in terms of combined safety rating and mileage."""
        cars = self.car_data.copy()
        # Custom standard score
        cars["OverallScore"] = (cars["SafetyRating"] * 10.0) + cars["Mileage"]
        return cars.sort_values(by="OverallScore", ascending=False).head(5)

    @staticmethod
    def _normalize_car_data(car_data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and standardizes the car dealer list.
        - Resolves spelling inconsistencies in headers.
        - Maps brand/model names to exact image file locations (resolving spaces and webp types).
        - Standardizes pricing to Lakhs (converts values like 12,00,000 to 12.0).
        """
        df = car_data.copy()

        # Rename columns to standard names
        aliases = {
            "Fuel": "FuelType",
            "Fuel Type": "FuelType",
            "Safety Rating": "SafetyRating",
            "Mileage (km/ltr)": "Mileage",
            "ExShowroomPrice": "Price",
            "Airbags": "Airbags",
            "Air Bags": "Airbags",
            "Safety Rating (NCAP)": "SafetyRating",
            "Buy Back": "BuyBackAvailable",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

        # Fill default columns if missing
        defaults = {
            "Brand": "Unknown",
            "Model": "Unknown",
            "Segment": "SUV",
            "FuelType": "Petrol",
            "Transmission": "Manual",
            "Price": 1000000.0,
            "Mileage": 15.0,
            "SeatingCapacity": 5,
            "SafetyRating": 4.0,
            "Airbags": 2,
            "BuyBackAvailable": "No",
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Clean NaN brand/model rows (these are spacers in the excel)
        df = df.dropna(subset=["Brand", "Model"])
        df = df[df["Brand"].astype(str).str.strip() != ""]

        # Parse numeric columns
        numeric_cols = ["Price", "Mileage", "SeatingCapacity", "SafetyRating", "Airbags"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])

        # IMPORTANT: Convert Ex-Showroom Price from Rupees (e.g. 12,00,000) to Lakhs (e.g. 12.0)
        df["Price"] = df["Price"].apply(lambda p: p / 100000.0 if p > 1000.0 else p)

        # Standardize car segment categories to match our standard predicted segments
        df["Segment"] = df["Segment"].astype(str).str.lower().str.strip()
        df["Segment"] = df["Segment"].map(CarRecommendationEngine.CAR_SEGMENT_MAP).fillna("SUV")

        # Standardize seating capacity & airbag counts to integer
        df["SeatingCapacity"] = df["SeatingCapacity"].astype(int)
        df["Airbags"] = df["Airbags"].astype(int)

        # Add clean Image filename column based on Brand, Model, and Segment
        import re
        import os
        def clean_filename(brand, model, segment):
            b = str(brand).lower().strip().replace(" ", "_").replace("-", "_")
            m = str(model).lower().strip().replace(" ", "_").replace("-", "_")
            
            # Handle "maruti" -> "maruti_suzuki" alias
            if b == "maruti":
                b = "maruti_suzuki"
            # Handle "hyundai" typo in Hyndai_Venue.webp
            if b == "hyundai" and m == "venue":
                return "Hyndai_Venue.webp"
            # Handle "mg hector.jpg" space typo
            if b == "mg" and m == "hector":
                return "mg hector.jpg"

            filename_base = f"{b}_{m}"
            
            # Try to find the file dynamically in the static/images/cars directory
            cars_dir = os.path.join("static", "images", "cars")
            if os.path.exists(cars_dir):
                for ext in [".jpg", ".webp", ".png", ".jpeg"]:
                    candidate = f"{filename_base}{ext}"
                    if os.path.exists(os.path.join(cars_dir, candidate)):
                        return candidate
                    # Check if just model exists
                    candidate_m = f"{m}{ext}"
                    if os.path.exists(os.path.join(cars_dir, candidate_m)):
                        return candidate_m

            # Segment fallback mappings if specific car image doesn't exist
            seg = str(segment).lower().strip()
            fallback_map = {
                "ev": "fallback_ev.jpg",
                "hatchback": "fallback_hatchback.jpg",
                "luxury": "fallback_luxury.jpg",
                "muv": "fallback_muv.jpg",
                "sedan": "fallback_sedan.jpg",
                "suv": "fallback_suv.jpg",
                "pickup truck": "fallback_suv.jpg"
            }
            return fallback_map.get(seg, "fallback_suv.jpg")
            
        # Standardize Dealer column and append registry details
        DEALER_REGISTRY = {
            "Aspa Bandsons Arena": {
                "Address": "Ghat Road, Near Cotton Market, Nagpur, Maharashtra 440018",
                "Phone": "+91 712 272 2221",
                "Email": "contact@aspabandsons.com"
            },
            "Aspa Auto NEXA": {
                "Address": "Kingsway Road, Near Railway Station, Nagpur, Maharashtra 440001",
                "Phone": "+91 712 666 3333",
                "Email": "nexa.kingsway@aspa.in"
            },
            "Jaika Motors": {
                "Address": "Civil Lines, Commercial Road, Nagpur, Maharashtra 440001",
                "Phone": "+91 712 663 8888",
                "Email": "sales@jaikamotors.co.in"
            },
            "Ketan Hyundai": {
                "Address": "Kamptee Road, Indora Chowk, Nagpur, Maharashtra 440017",
                "Phone": "+91 712 264 5555",
                "Email": "info@ketanhyundai.com"
            },
            "JPS Kia": {
                "Address": "Great Nag Road, Baidyanath Chowk, Nagpur, Maharashtra 440009",
                "Phone": "+91 712 270 4444",
                "Email": "support@jpskia.com"
            },
            "Murli Toyota": {
                "Address": "Amravati Road, Wadi, Nagpur, Maharashtra 440023",
                "Phone": "+91 712 281 1111",
                "Email": "contact@murlitoyota.co.in"
            },
            "Girnar Honda": {
                "Address": "Central Avenue, Nagpur, Maharashtra 440008",
                "Phone": "+91 712 273 9999",
                "Email": "sales@girnarhonda.com"
            },
            "Renault Amravati": {
                "Address": "Badnera Road, Amravati, Maharashtra 444605",
                "Phone": "+91 721 251 3333",
                "Email": "amravati@renault-india.com"
            },
            "MG Motor Nangia": {
                "Address": "Hingna Road, Nagpur, Maharashtra 440016",
                "Phone": "+91 712 668 5555",
                "Email": "nangia.mg@mginbound.com"
            }
        }
        
        default_contact = {
            "Address": "Main Automotive Hub, Civil Lines, Nagpur, Maharashtra 440001",
            "Phone": "+91 1800 200 4455",
            "Email": "partners@drivewise.com"
        }

        if "Dealer" not in df.columns:
            df["Dealer"] = "DriveWise Partner Dealer"
        else:
            df["Dealer"] = df["Dealer"].fillna("DriveWise Partner Dealer").astype(str).str.strip()

        df["DealerAddress"] = df["Dealer"].apply(lambda d: DEALER_REGISTRY.get(d, default_contact)["Address"])
        df["DealerPhone"] = df["Dealer"].apply(lambda d: DEALER_REGISTRY.get(d, default_contact)["Phone"])
        df["DealerEmail"] = df["Dealer"].apply(lambda d: DEALER_REGISTRY.get(d, default_contact)["Email"])
            
        df["Image"] = df.apply(lambda r: clean_filename(str(r["Brand"]), str(r["Model"]), str(r["Segment"])), axis=1)

        return df