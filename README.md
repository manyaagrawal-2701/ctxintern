# DriveWise AI: Personal Vehicle Matchmaker & Dealer Routing SaaS

DriveWise is a secure, intelligent automotive recommendation and dealership routing platform. It maps consumer demographic profiles (age, income limits, budget boundaries, and family size) to their statistically ideal vehicle segment using an **XGBoost machine learning classifier**, then scores and ranks local dealership catalogs to recommend the top matches.

---

## 🌟 Core Features

* **Demographic Classification (AI)**: Employs a regularized XGBoost classifier (88.33% F1-Score) to predict the suitable segment category based on household demographics.
* **100-Point Proximity Matching**: Ranks cars based on budget boundaries, fuel efficiency, seating requirements, safety ratings, and transmission matching.
* **Segment Diversity Filters**: Enforces a diversity limit of at most 2 recommendations per vehicle segment (e.g. MUV, SUV, EV) to offer a balanced choice.
* **Dealership Contact Routing**: Locates and renders exact dealership addresses, direct phone lines, and email listings for in-stock models.
* **Multi-Tenant Security**: Protects user logs, logins, and passwords using `werkzeug.security` (PBKDF2-SHA256) hash signatures and `flask.session` guards.
* **Dynamic Analytics Dashboards**: Automatically refreshes Matplotlib visualizations (Segment, Brand, Fuel, Budget, and Transmission distributions) isolated by user ID to prevent concurrent image cache conflicts.
* **Auditing & Reporting**: Generates automated Excel catalogs (`training_data_report.xlsx`) and executive PDF summaries (`customer_analysis_report.pdf`) during retraining runs.

---

## 📂 Project Architecture

```
CarRecommendationSystem/
├── config.py                 # Dynamic directory pathing & server settings
├── app.py                    # Flask web controller (routes and session handlers)
├── train.py                  # Pipeline orchestrator for model tournament retraining
├── core/
│   ├── dataloader.py         # File loader supporting CSV/XLSX & regex header locator
│   ├── preprocessing.py      # Feature engineering (ratios, slabs, and OHE/scaling)
│   ├── model_trainer.py      # Model comparisons, GridSearchCV, and XGBoost wrappers
│   ├── recommendation_engine.py # Proximity matchmaker and dealership registry
│   ├── report_generator.py   # Non-interactive PDF reports & Matplotlib visualizers
│   └── database_manager.py   # MySQL schema setups, password hashing, and user logs
├── static/
│   ├── cssstyle.css          # Vanilla custom CSS styling
│   └── images/
│       └── cars/             # Vehicle display photos and fallback segment images
├── templates/                # Bootstrap Jinja HTML view files
├── tests/                    # Recommendation engine test suites
├── requirements.txt          # Python library dependency mappings
└── README.md                 # Project handbook
```

---

## 🚀 Local Installation & Setup (Windows)

### Prerequisites
* Python 3.11 or later installed.
* MySQL Server database running locally.

### 1. Clone & Set Up Virtual Environment
Open PowerShell inside the project directory:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Database
Ensure MySQL Server is active. By default, the app looks for root credentials. Update your port and password in [`config.py`](file:///d:/Projects/CarRecommendationSystem/config.py#L25-L35) if necessary.

---

## 🏋️ Model Training & Evaluation
To run the preprocessor transformations, execute the GridSearchCV hyperparameter tournament, and generate baseline PDF reports:
```powershell
python train.py
```
This selects the best model (XGBoost), saves serialized binaries (`best_model.pkl` & `preprocessor.pkl`) inside the `models/` directory, and outputs performance logs.

---

## 🌐 Launching the Web Server
Launch the Flask development server:
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your web browser. 
1. Register a new user profile or log in.
2. Navigate to the matching form to calculate recommendations.
3. View matching details, interactive dealership modals, and user-isolated analytics dashboards.

---

## 🧪 Running Unit Tests
To execute the test suites:
```powershell
python -m pytest -o pythonpath=.
```
