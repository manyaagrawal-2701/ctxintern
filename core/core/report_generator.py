# ==============================================================================
# CLASS: ReportGenerator
# PURPOSE: Generates analytical reports (Excel files, multi-page PDFs) 
#          and creates visual dashboard plots (Bar charts, Scatter plots, Pie charts).
# ==============================================================================

import logging
from pathlib import Path
import matplotlib
# Use a non-interactive backend (Agg) to prevent Flask web server threads
# from trying to open GUI popup windows when generating plots in the background.
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from config import Config

# Logger to track pdf/excel file exports
LOGGER = logging.getLogger(__name__)


class ReportGenerator:
    """Generates Excel, PDF reports and saves static visualization charts."""

    def __init__(self, report_dir=Config.REPORT_DIR) -> None:
        """
        Constructor. Creates report export directories and sets the SeaBorn plot style.
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        # Use SeaBorn's clean modern grid styling for charts
        sns.set_theme(style="whitegrid")

    def generate_excel_report(self, df: pd.DataFrame, filename: str = "training_data_report.xlsx") -> Path:
        """
        Generates a multi-sheet spreadsheet report analyzing the dataset.
        - Sheet 1: The main dataset.
        - Sheet 2: Descriptive stats (mean, min, max, standard deviations).
        - Sheet 3: A checklist of missing (null) values.
        - Sheet 4: Column data types.
        """
        filepath = self.report_dir / filename

        try:
            # openpyxl engine allows creating complex multi-sheet Excel files
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # 1. Main Sheet
                df.to_excel(writer, sheet_name="Dataset", index=False)

                # 2. Summary stats sheet (transposed so metrics list vertically)
                summary = df.describe(include="all").transpose()
                summary.to_excel(writer, sheet_name="Summary")

                # 3. Missing slots audit sheet
                missing = pd.DataFrame({
                    "Missing Values": df.isnull().sum(),
                    "Percentage (%)": round(df.isnull().mean() * 100, 2)
                })
                missing.to_excel(writer, sheet_name="Missing Values")

                # 4. Column Data Types sheet
                datatype = pd.DataFrame({
                    "Column": df.columns,
                    "Datatype": df.dtypes.astype(str)
                })
                datatype.to_excel(writer, sheet_name="Data Types", index=False)

            LOGGER.info("Excel Report Saved: %s", filepath)
            return filepath
        except Exception as e:
            LOGGER.exception("Failed to generate Excel report.")
            raise e

    def generate_pdf_report(self, df: pd.DataFrame, filename: str = "customer_analysis_report.pdf") -> Path:
        """
        Generates a professional 2-page PDF summary report using Matplotlib PdfPages:
        - Page 1: Cover page with executive summary text and segment distributions.
        - Page 2: A 2x2 grid of data distribution charts (histograms, pie charts, scatter plots).
        """
        filepath = self.report_dir / filename

        try:
            with PdfPages(filepath) as pdf:
                # -----------------------------------------------------
                # PAGE 1: TITLE & SUMMARY TEXT
                # -----------------------------------------------------
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")  # Turn off border ticks and grids to draw text page

                # Header Title Banner
                ax.text(0.5, 0.90, "Car Recommendation System", fontsize=24, weight="bold", ha="center", color="#1e3a8a")
                ax.text(0.5, 0.85, "Customer Insights & Demographic Report", fontsize=14, ha="center", color="#4b5563")
                ax.axhline(0.82, 0.1, 0.9, color="#1e3a8a", linewidth=2)

                # Paragraph 1: Executive Summary
                ax.text(0.1, 0.75, "1. Executive Summary", fontsize=16, weight="bold", color="#1e293b")
                summary_text = (
                    f"This report presents an analytical summary of the historical customer database.\n"
                    f"The dataset consists of demographic records, financial indicators, and car preference\n"
                    f"histories, which are used to train the machine learning classification models.\n\n"
                    f"• Total Customer Records Analyzed: {len(df)}\n"
                    f"• Average Customer Age: {df['Age'].mean():.1f} years\n"
                    f"• Median Monthly Income Slab: ₹{pd.to_numeric(df.get('MonthlyIncome', 0), errors='coerce').fillna(df.get('Min_Monthly_Income', 0)).median():,.2f}\n"
                    f"• Average Car Purchasing Budget: ₹{df['Budget'].mean():.2f} Lakhs"
                )
                ax.text(0.1, 0.60, summary_text, fontsize=11, linespacing=1.6)

                # Paragraph 2: Target Segment count distributions
                ax.text(0.1, 0.50, "2. Segment Class Distributions", fontsize=16, weight="bold", color="#1e293b")
                seg_counts = df["TargetCarSegment"].value_counts()
                seg_text = "\n".join([f"  • {seg}: {count} customers ({count/len(df)*100:.1f}%)" for seg, count in seg_counts.items()])
                ax.text(0.1, 0.35, seg_text, fontsize=11, linespacing=1.6)

                # Footer Note
                ax.text(0.5, 0.05, "Generated automatically by Car Recommendation System backend.", fontsize=9, ha="center", style="italic", color="#9ca3af")

                plt.tight_layout()
                pdf.savefig(fig)  # Saves this figure as Page 1
                plt.close(fig)

                # -----------------------------------------------------
                # PAGE 2: STATISTICAL CHART PORTFOLIO (2x2 Grid)
                # -----------------------------------------------------
                fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
                
                # Plot 2.1: Age distribution histogram
                sns.histplot(df["Age"], bins=15, kde=True, ax=axes[0, 0], color="#2563eb")
                axes[0, 0].set_title("Age Distribution", fontsize=12, weight="bold")
                axes[0, 0].set_xlabel("Age")

                # Plot 2.2: Target Vehicle Segment bar count
                sns.countplot(data=df, x="TargetCarSegment", ax=axes[0, 1], palette="Blues_r")
                axes[0, 1].set_title("Segment Distribution", fontsize=12, weight="bold")
                axes[0, 1].set_xlabel("Segment")
                axes[0, 1].tick_params(axis="x", rotation=30)

                # Plot 2.3: Fuel Preference shares pie chart
                fuel_counts = df["FuelPreference"].value_counts()
                axes[1, 0].pie(fuel_counts, labels=fuel_counts.index, autopct="%1.1f%%", colors=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
                axes[1, 0].set_title("Fuel Preference", fontsize=12, weight="bold")

                # Plot 2.4: Annual Income vs Car Budget scatter plot
                sns.scatterplot(data=df, x="AnnualIncome", y="Budget", ax=axes[1, 1], color="#10b981", alpha=0.6)
                axes[1, 1].set_title("Income vs Budget", fontsize=12, weight="bold")
                axes[1, 1].set_xlabel("Annual Income (₹)")
                axes[1, 1].set_ylabel("Budget (₹ Lakh)")

                plt.suptitle("Statistical Chart Portfolio", fontsize=16, weight="bold", color="#1e3a8a", y=0.98)
                plt.tight_layout()
                pdf.savefig(fig)  # Saves this grid as Page 2
                plt.close(fig)

            LOGGER.info("PDF Report Saved: %s", filepath)
            return filepath
        except Exception as e:
            LOGGER.exception("Failed to generate PDF report.")
            raise e

    def create_visualizations(self, df: pd.DataFrame, user_id: int | None = None) -> dict:
        """
        Generates individual high-resolution PNG charts from database logs.
        Saves them to the 'static/images/' folder so they can load on the Web Dashboard:
        1. segment_distribution.png: Counts of predicted segments.
        2. brand_preference.png: Most commonly recommended car brands.
        3. fuel_preference.png: Customer fuel preference distribution.
        4. budget_distribution.png: Customer budgets spread.
        5. income_vs_budget.png: Income vs budget scatter plot.
        6. correlation_heatmap.png: Heatmap grid showing correlations between columns.
        """
        charts = {}
        if df.empty:
            return charts

        suffix = f"_{user_id}" if user_id is not None else ""

        # Ensure output directory exists in static/images
        img_dir = Config.BASE_DIR / "static" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Segment Distribution Bar Chart
            seg_col = "predicted_segment" if "predicted_segment" in df.columns else ("TargetCarSegment" if "TargetCarSegment" in df.columns else None)
            if seg_col:
                plt.figure(figsize=(6, 4))
                # x variable is assigned to hue and legend=False is set to follow seaborn deprecation rules
                sns.countplot(data=df, x=seg_col, hue=seg_col, palette="viridis", legend=False)
                plt.title("Vehicle Segment Distribution", fontsize=12, weight="bold")
                plt.xlabel("Segment")
                plt.ylabel("Count")
                plt.tight_layout()
                path = img_dir / f"segment_distribution{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["segment_distribution"] = path

            # 2. Recommended Brands Count Bar Chart
            brand_col = "recommended_brand" if "recommended_brand" in df.columns else ("Brand" if "Brand" in df.columns else None)
            if brand_col and brand_col in df.columns:
                plt.figure(figsize=(6, 4))
                df[brand_col].value_counts().head(8).plot(kind="bar", color="#0d6efd")
                plt.title("Recommended Brand Preference", fontsize=12, weight="bold")
                plt.xlabel("Brand")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = img_dir / f"brand_preference{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["brand_preference"] = path

            # 3. Fuel Preference Share Pie Chart
            fuel_col = "fuel_preference" if "fuel_preference" in df.columns else ("FuelPreference" if "FuelPreference" in df.columns else None)
            if fuel_col and fuel_col in df.columns:
                plt.figure(figsize=(5, 4))
                df[fuel_col].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=sns.color_palette("pastel"))
                plt.ylabel("")
                plt.title("Fuel Preference Distribution", fontsize=12, weight="bold")
                plt.tight_layout()
                path = img_dir / f"fuel_preference{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["fuel_preference"] = path

            # 4. Budget Range Histogram
            budget_col = "budget" if "budget" in df.columns else ("Budget" if "Budget" in df.columns else None)
            if budget_col and budget_col in df.columns:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[budget_col], bins=10, color="#198754", kde=True)
                plt.title("Customer Budget Distribution", fontsize=12, weight="bold")
                plt.xlabel("Budget (₹ Lakh)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                path = img_dir / f"budget_distribution{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["budget_distribution"] = path

            # 5. Income vs Budget Scatter Plot
            income_col = "AnnualIncome" if "AnnualIncome" in df.columns else None
            budget_col = "Budget" if "Budget" in df.columns else None
            if income_col and budget_col:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=df, x=income_col, y=budget_col, hue="TargetCarSegment", palette="Set2")
                plt.title("Annual Income vs Car Budget", fontsize=12, weight="bold")
                plt.xlabel("Annual Income (₹)")
                plt.ylabel("Budget (₹ Lakh)")
                plt.tight_layout()
                path = img_dir / f"income_vs_budget{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["income_vs_budget"] = path

            # 6. Correlation Grid Heatmap
            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                plt.figure(figsize=(6, 4))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                plt.title("Feature Correlation Heatmap", fontsize=12, weight="bold")
                plt.tight_layout()
                path = img_dir / f"correlation_heatmap{suffix}.png"
                plt.savefig(path, dpi=120)
                plt.close()
                charts["correlation_heatmap"] = path

            LOGGER.info("Dashboard visualizations generated successfully.")
            return charts
        except Exception as e:
            LOGGER.exception("Failed to generate dashboard visualizations.")
            return charts