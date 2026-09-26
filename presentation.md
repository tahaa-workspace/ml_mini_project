# Presentation — House Rent Prediction

## Slide 1 — Title
**House Rent Prediction Using Machine Learning**
MC03094121 Machine Learning
MCA Department, SVIT Vasad

## Slide 2 — Problem Statement
Predict monthly house rent from property characteristics such as size, BHK, bathrooms, city, furnishing status, floor, and tenant preference.

## Slide 3 — Objectives
- Understand a real-world regression problem
- Clean and preprocess mixed tabular data
- Perform EDA
- Select meaningful features
- Compare four regression algorithms
- Evaluate models using MAE, MSE, RMSE and R²
- Build a simple prediction demo

## Slide 4 — Dataset
- 4,746 records
- 12 columns
- Target: Rent
- Numerical + categorical data
- No missing values
- No duplicate records
- Six cities represented

## Slide 5 — Data Preprocessing
- Parsed Floor into Floor Number and Total Floors
- Corrected two logically inconsistent floor records to missing parsed values
- Extracted month and day-of-week from Posted On
- One-hot encoded categorical features
- Standardized numerical features
- Did not automatically remove high-rent observations

## Slide 6 — EDA
Include:
- Rent histogram
- City bar chart
- Size vs Rent scatter plot
- Rent box plot
- Correlation heatmap

## Slide 7 — Feature Selection
Selected:
BHK, Size, Floor Number, Total Floors, Area Type, City, Furnishing Status, Tenant Preferred, Bathroom, Posted Month, Posted Day of Week.

Removed:
- Point of Contact — not a property-price feature
- Area Locality — 2,235 unique values, too high-cardinality for this basic project
- Raw Floor / Posted On — transformed

## Slide 8 — ML Algorithms
- Linear Regression — baseline
- Decision Tree Regressor — non-linear rules
- Random Forest Regressor — ensemble trees
- Gradient Boosting Regressor — sequential boosting

## Slide 9 — Model Training
Dataset → Preprocessing → 80/20 Train-Test Split → Training → Prediction → Evaluation

X = input features
y = Rent
fit() = learn from training data
predict() = estimate rent for test/new data

## Slide 10 — Results & Evaluation
| Algorithm | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | ₹21,903 | 1,912,068,304 | ₹43,727 | 0.520 |
| Decision Tree Regressor | ₹14,189 | 1,660,077,977 | ₹40,744 | 0.583 |
| Random Forest Regressor | ₹12,469 | 1,317,987,514 | ₹36,304 | 0.669 |
| Gradient Boosting Regressor | ₹11,242 | 1,353,387,291 | ₹36,788 | 0.660 |

Selected model: **Random Forest Regressor**, based on the strongest R² (0.669) and lowest RMSE (₹36,304) in this experiment.

Note: Gradient Boosting has the lowest MAE (₹11,242), so model choice depends on the evaluation priority.

## Slide 11 — Conclusion & Future Scope
Conclusion:
- The project successfully implements an end-to-end regression workflow.
- Random Forest was selected using R²/RMSE as the primary criteria.

Future scope:
- Cross-validation
- Hyperparameter tuning
- Better locality encoding
- More geographic/property features
- Larger and newer datasets
- Rent transformation to handle skew

## Slide 12 — Demo
Show the Streamlit app:
Input property details → Click Predict Rent → Display predicted monthly rent.
