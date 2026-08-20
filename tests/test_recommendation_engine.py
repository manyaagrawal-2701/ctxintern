import pandas as pd

from core.recommendation_engine import CarRecommendationEngine


def test_recommend_car_filters_by_segment_and_budget():
    cars = pd.DataFrame(
        [
            {
                "Brand": "Hyundai",
                "Model": "Creta",
                "Segment": "SUV",
                "FuelType": "Petrol",
                "Transmission": "Manual",
                "Price": 12,
                "Mileage": 17,
                "SafetyRating": 4.5,
                "SeatingCapacity": 5,
            },
            {
                "Brand": "Mercedes-Benz",
                "Model": "C-Class",
                "Segment": "Luxury",
                "FuelType": "Petrol",
                "Transmission": "Automatic",
                "Price": 58,
                "Mileage": 16,
                "SafetyRating": 5,
                "SeatingCapacity": 5,
            },
        ]
    )
    engine = CarRecommendationEngine(model=None, car_data=cars)

    recommendations = engine.recommend_car("SUV", budget=15)

    assert recommendations.iloc[0]["Brand"] == "Hyundai"
