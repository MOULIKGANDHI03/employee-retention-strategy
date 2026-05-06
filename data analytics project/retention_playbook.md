# Employee Retention Playbook
I ran some ML models on the IBM HR dataset to see what drives people to quit. Here's what I found.

## Model Stats
- **Logistic Regression Accuracy:** 86.1%
- **Random Forest Accuracy:** 83.3%

## Top 3 Drivers of Attrition
Based on the Random Forest feature importances, the biggest factors are:

1. **MonthlyIncome** (Score: 0.072)
2. **Age** (Score: 0.072)
3. **TotalWorkingYears** (Score: 0.054)

## EDA Notes
From looking at the charts I generated:
- **Overtime:** People working overtime quit way more often.
- **Income:** People making less money are definitely more at risk of leaving.
- **Tenure:** Newer people quit more than veterans.

## What should HR do?

### 1. Look at MonthlyIncome
*   If this is income, we probably need to review pay bands and make sure we're competitive.
*   Maybe throw in some spot bonuses for critical roles.

### 2. Look at Age
*   If this relates to overtime or age, we need to make sure managers aren't burning out their teams.
*   Enforce hard cut-offs for slacks/emails after 6 PM.

### 3. Look at TotalWorkingYears
*   If this is tenure/years working, we need a better onboarding buddy system so people feel connected in their first year.

*Note: Replacing someone costs like 2x their salary, so fixing even a few of these saves the company a ton of money.*
