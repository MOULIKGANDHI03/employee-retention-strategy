# Employee Attrition & Retention Strategy

I built this end-to-end data analytics pipeline to explore HR analytics and understand the core drivers of employee turnover. High turnover is incredibly expensive (costing roughly 1.5x to 2x an employee's salary to replace them), so the goal of this project is to identify "at-risk" employees using machine learning and provide actionable retention strategies.

## Project Features

This project serves as a complete Minimum Viable Product (MVP) for an HR analytics workflow:
1. **Data Ingestion & Cleaning:** Processes raw IBM HR data, handles categorical encoding, and standardizes numerical features.
2. **Exploratory Data Analysis (EDA):** Generates insightful visualizations to uncover links between employee tenure, salary, overtime, and attrition rates.
3. **Machine Learning Models:** Trains and evaluates both **Logistic Regression** and **Random Forest** classification models to predict attrition.
4. **Model Interpretability:** Extracts feature importances and coefficients to identify exactly *why* the models make their decisions. It generates color-coded feature importance charts (highlighting risk factors vs. retention factors).
5. **Actionable Business Value:** Automatically generates a `retention_playbook.md` that translates raw mathematical insights into practical advice for HR teams.

## Tech Stack
- **Python**
- **pandas, numpy** for data wrangling
- **scikit-learn** for machine learning pipelines
- **matplotlib, seaborn** for data visualization

## The Data
I used the IBM HR Analytics Employee Attrition & Performance dataset. 

## How to Run It Locally
1. Make sure Python is installed on your system.
2. Run `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Make sure the dataset `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in this folder.
4. Run the analysis script:
   ```bash
   python attrition_analysis.py
   ```

After running the script, the project will:
- Clean and process the dataset.
- Train machine learning models.
- Generate visual charts in .png format.
- Create a retention_playbook.md file with HR insights and employee retention recommendations.

## Current Results & Insights
- The **Logistic Regression** model achieved ~86% accuracy, successfully identifying factors that push employees toward or away from quitting.
- The **Random Forest** model achieved ~83% accuracy.
- Across the models, the top 3 drivers for employee turnover were identified as **Monthly Income**, **Age**, and **Total Working Years**. Lower income and early-career status heavily correlate with higher attrition risk.

## Future Improvements (Next Steps)
While this pipeline is highly functional, future iterations could include:
- **Handling Class Imbalance:** Implementing techniques like SMOTE (Synthetic Minority Over-sampling Technique) since the dataset is naturally imbalanced (~84% retention rate).
- **Advanced Evaluation Metrics:** Adding Confusion Matrices and ROC-AUC curves to better track Precision and Recall.
- **Interactive Dashboard:** Building a Streamlit web app to allow HR managers to input employee statistics and receive an instant "Flight Risk %" prediction.
