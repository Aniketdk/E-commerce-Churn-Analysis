# E-commerce Customer Churn Prediction Project 🛒
This project focuses on building a machine learning model to predict which e-commerce customers are likely to churn (stop purchasing) and identifying the key behavioral drivers behind that attrition. The goal is to provide actionable insights for targeted customer retention campaigns.

#🚀 Project OverviewCustomer retention is vital for e-commerce profitability. 
By using historical transactional and behavioral data, this project aims to:
Identify High-Risk Customers.
Use a Random Forest classifier to assign a Churn Probability Score to all active customers.
Uncover Key Drivers: Determine which factors (e.g., Recency, Order Frequency, Subscription Status) have the highest influence on churn.
Provide Actionable Data: Output a scored dataset ready for business intelligence tools like Power BI or Tableau to guide retention efforts.


#🛠️ Technologies & LibrariesLanguage: Python 3.xData Manipulation: pandas, numpyMachine Learning: scikit-learn (for Random Forest Classifier, StandardScaler, train_test_split)Visualization (In Analysis): matplotlib, seabornBusiness Intelligence: Power BI (for dashboard visualization)

#⚙️ Getting StartedFollow these steps to set up the project locally and run the analysis.
1. Clone the RepositoryBashgit clone https://github.com/YourUsername/E-commerce-Churn-Prediction.git
cd E-commerce-Churn-Prediction
2. Install DependenciesIt's recommended to use a virtual environment.Bashpip install -r requirements.txt
(Note: Create a requirements.txt file listing the necessary libraries: pandas, numpy, scikit-learn, matplotlib, seaborn)
3. Generate DataThis project uses a synthetic dataset.
Run the data generator script to create the initial CSV file:Bashpython src/data_generator.py
This command creates data/ecommerce_data.csv.

#🔑 Key Findings & Business InsightsThe Random Forest model typically highlights the following factors as the most significant drivers of churn in this type of e-commerce data:
Recency_Days: Customers who haven't purchased in a long time are the highest risk.
Orders_Last_6M: Low order frequency is a strong indicator of disengagement.
Subscription Status: Customers without a premium subscription have a significantly higher churn rate.


#📈 Power BI DashboardThe final scored CSV (ecommerce_churn_scored_for_BI.csv) is the input for the Power BI dashboard. 
The dashboard visualizes:
High-Risk Customer Segments (for immediate action).
Churn Rate by Recency and Frequency Buckets.
Feature Importance ranking to confirm business hypothesis.
