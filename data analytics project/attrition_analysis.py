import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# folder setup
BASE_DIR = r"d:\project\data analytics project"
DATA_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
PLAYBOOK_FILE = os.path.join(BASE_DIR, "retention_playbook.md")

def main():
    print("loading up the data...")
    df = pd.read_csv(DATA_FILE)
    
    print("running eda and making some charts...")
    sns.set_theme(style="whitegrid")
    
    # overtime vs attrition chart
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="OverTime", hue="Attrition", palette="Set2")
    plt.title("Attrition by OverTime")
    plt.savefig(os.path.join(BASE_DIR, "attrition_by_overtime.png"))
    plt.close()
    
    # income chart
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", palette="Set1")
    plt.title("Monthly Income vs Attrition")
    plt.savefig(os.path.join(BASE_DIR, "attrition_by_income.png"))
    plt.close()
    
    # tenure chart
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Attrition", y="YearsAtCompany", palette="Set3")
    plt.title("Years At Company vs Attrition")
    plt.savefig(os.path.join(BASE_DIR, "attrition_by_tenure.png"))
    plt.close()

    print("cleaning data...")
    # drop stuff we don't need
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # target var
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(columns=['Attrition'])
    
    # encode categories
    cat_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # scale data for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("training models now...")
    
    # logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"LogReg Accuracy: {lr_acc:.4f}")
    
    # logistic regression feature importance chart
    print("generating logistic regression feature importance chart...")
    lr_coeffs = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Coefficient': lr.coef_[0]
    })
    lr_coeffs['Abs_Coefficient'] = lr_coeffs['Coefficient'].abs()
    top_lr_coeffs = lr_coeffs.sort_values(by='Abs_Coefficient', ascending=False).head(15)
    
    # color by direction
    top_lr_coeffs['Impact'] = np.where(top_lr_coeffs['Coefficient'] > 0, 'Increases Attrition', 'Decreases Attrition')
    
    plt.figure(figsize=(10, 8))
    # using Set2 colors: #fc8d62 (orange) for Yes/Increases, #66c2a5 (green) for No/Decreases
    sns.barplot(data=top_lr_coeffs, x='Coefficient', y='Feature', hue='Impact', 
                palette={'Increases Attrition': '#fc8d62', 'Decreases Attrition': '#66c2a5'}, dodge=False)
    plt.title('Top 15 Logistic Regression Feature Coefficients')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "lr_feature_importance.png"))
    plt.close()

    
    # random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) 
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    print("pulling top features from rf...")
    feature_importances = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("generating random forest feature importance chart...")
    top_15_rf = feature_importances.head(15)
    plt.figure(figsize=(10, 8))
    # using the Set2 Attrition (Yes) color for the random forest chart
    sns.barplot(data=top_15_rf, x='Importance', y='Feature', color='#fc8d62')
    plt.title('Top 15 Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "rf_feature_importance.png"))
    plt.close()
    
    top_3 = feature_importances.head(3)
    print("\nTop 3 drivers:")
    print(top_3.to_string(index=False))
    
    print("writing playbook...")
    
    playbook_content = f"""# Employee Retention Playbook
I ran some ML models on the IBM HR dataset to see what drives people to quit. Here's what I found.

## Model Stats
- **Logistic Regression Accuracy:** {lr_acc:.1%}
- **Random Forest Accuracy:** {rf_acc:.1%}

## Top 3 Drivers of Attrition
Based on the Random Forest feature importances, the biggest factors are:

1. **{top_3.iloc[0]['Feature']}** (Score: {top_3.iloc[0]['Importance']:.3f})
2. **{top_3.iloc[1]['Feature']}** (Score: {top_3.iloc[1]['Importance']:.3f})
3. **{top_3.iloc[2]['Feature']}** (Score: {top_3.iloc[2]['Importance']:.3f})

## EDA Notes
From looking at the charts I generated:
- **Overtime:** People working overtime quit way more often.
- **Income:** People making less money are definitely more at risk of leaving.
- **Tenure:** Newer people quit more than veterans.

## What should HR do?

### 1. Look at {top_3.iloc[0]['Feature']}
*   If this is income, we probably need to review pay bands and make sure we're competitive.
*   Maybe throw in some spot bonuses for critical roles.

### 2. Look at {top_3.iloc[1]['Feature']}
*   If this relates to overtime or age, we need to make sure managers aren't burning out their teams.
*   Enforce hard cut-offs for slacks/emails after 6 PM.

### 3. Look at {top_3.iloc[2]['Feature']}
*   If this is tenure/years working, we need a better onboarding buddy system so people feel connected in their first year.

*Note: Replacing someone costs like 2x their salary, so fixing even a few of these saves the company a ton of money.*
"""
    
    with open(PLAYBOOK_FILE, "w") as f:
        f.write(playbook_content)
        
    print(f"done! check out {PLAYBOOK_FILE}")

if __name__ == "__main__":
    main()
