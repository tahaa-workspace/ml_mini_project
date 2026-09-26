# House Rent Prediction — Machine Learning Project

## 1. Problem Definition

### Real-world problem
House rent varies with property size, number of bedrooms and bathrooms, city, furnishing status, floor, and tenant preferences. The project builds a machine-learning regression system that predicts the expected monthly rent of a listed property from its available characteristics.

### Why the problem is important
A rent estimate can support tenants, property owners, and rental platforms when comparing or assessing properties.

### Beneficiaries
- Tenants comparing rental options
- Property owners/agents estimating a listing price
- Rental platforms that need automated price-estimation support

### Expected output
A predicted monthly rent value for a new property.

## 2. Dataset Understanding

The uploaded dataset is `House_Rent_Dataset.csv`.

- Rows: **4746**
- Columns: **12**
- Target variable: **Rent**
- Problem type: **Regression**

The glossary describes BHK as bedrooms/hall/kitchen, Rent as the property rent, Size as square feet, Floor as current floor and total floors, Area Type as Super/Carpet/Built Area, Area Locality as locality, City as city, Furnishing Status as furnished/semi-furnished/unfurnished, Tenant Preferred as preferred tenant type, Bathroom as number of bathrooms, and Point of Contact as the contact party.

### Data types
Numerical: BHK, Rent, Size, Bathroom

Categorical/textual: Posted On, Floor, Area Type, Area Locality, City, Furnishing Status, Tenant Preferred, Point of Contact

### Data quality
- Missing values: **0**
- Duplicate records: **0**
- Unique localities: **2235**

## 3. Data Preprocessing

1. **Missing values:** No original missing values were present. Median imputation is retained in the numerical preprocessing pipeline as a safety mechanism for values created during parsing.
2. **Duplicates:** No duplicate rows were found.
3. **Incorrect/structured Floor data:** `Floor` values such as "Ground out of 2" were split into `Floor_Number` and `Total_Floors`. Two logically inconsistent records ("8 out of 5" and "2 out of 1") were converted to missing parsed floor values and subsequently median-imputed.
4. **Date processing:** `Posted On` was converted to date and represented by month and day-of-week.
5. **Categorical encoding:** Area Type, City, Furnishing Status, and Tenant Preferred use One-Hot Encoding.
6. **Scaling:** Numerical features are standardized in the common preprocessing pipeline. This is important for Linear Regression and does not prevent tree models from working.
7. **Outliers:** Rent has a long right tail and IQR analysis flags high-value observations. They were **not automatically removed**, because they may represent genuine expensive rental properties; removing them could bias the model.

### Why these steps?
The objective is to convert mixed real-world tabular data into a consistent numerical representation while preserving meaningful property information.

## 4. Exploratory Data Analysis

The project includes:
- Histogram — distribution of Rent
- Bar chart — listing distribution by City
- Scatter plot — Size vs Rent relationship
- Box plot — Rent outliers/spread
- Correlation heatmap — numerical relationships

Selected observations:
- Mumbai has the highest mean rent among the six cities in this dataset.
- Size and Bathroom have positive correlations with Rent.
- Rent has substantial right-skew due to high-rent listings.

## 5. Feature Selection

### Selected features
BHK, Size, Floor_Number, Total_Floors, Area Type, City, Furnishing Status, Tenant Preferred, Bathroom, Posted_Month, Posted_DayOfWeek.

### Removed features
- **Point of Contact:** operational/contact information rather than a property-price characteristic.
- **Area Locality:** 2,235 unique values across 4,746 records creates very high cardinality for a basic subject-level project. Keeping it would create a large sparse feature space and complicate interpretation.
- Raw **Floor** and raw **Posted On** were transformed into more useful features.

If irrelevant or excessively high-cardinality information is included without care, the model can become more complex, less interpretable, and more susceptible to noisy/sparse representations.

## 6. ML Algorithms

Four regression algorithms were compared:

1. Linear Regression — baseline linear relationship.
2. Decision Tree Regressor — captures non-linear decision rules.
3. Random Forest Regressor — ensemble of decision trees; robust for non-linear tabular data.
4. Gradient Boosting Regressor — sequentially improves predictions by learning from previous errors.

## 7. Model Training

Workflow:

**Dataset → Preprocessing → Train/Test Split → Training → Prediction → Evaluation**

- X = selected input features
- y = Rent
- Test size = 20%
- Training size = 80%
- Random state = 42

`fit()` learns model parameters from training data. `predict()` generates rent estimates for unseen test examples.

## 8. Model Evaluation

For regression, the guideline requires MAE, MSE, RMSE and R².

| Algorithm | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | ₹21,902.57 | 1,912,068,304.49 | ₹43,727.20 | 0.5202 |\n| Decision Tree Regressor | ₹14,189.13 | 1,660,077,976.57 | ₹40,744.05 | 0.5835 |\n| Random Forest Regressor | ₹12,469.31 | 1,317,987,513.52 | ₹36,304.10 | 0.6693 |\n| Gradient Boosting Regressor | ₹11,242.49 | 1,353,387,291.12 | ₹36,788.41 | 0.6604 |\n
## 9. Results and Interpretation

Using R² and RMSE as the primary comparison criteria, **Random Forest Regressor** is selected for the project:
- R² = **0.6693**
- RMSE = **₹36,304.10**
- MAE = **₹12,469.31**

Gradient Boosting achieved a slightly lower MAE (**₹11,242.49**) than Random Forest, so the comparison is not uniformly won by one metric. Random Forest has the stronger R² and RMSE in this run.

### Important feature signals
The fitted Random Forest gives high importance to Size, Bathroom, and City-related encoded features. Feature importance indicates predictive contribution within this fitted model; it should not be interpreted as proof of causation.

### Limitations
- The dataset covers only the listings contained in the supplied file.
- Area Locality was removed because of very high cardinality, so fine-grained locality effects are not fully represented.
- Rental prices can change over time and may depend on factors not present in the dataset.
- A single train/test split provides only one evaluation sample.
- Tree models can still be affected by unusual high-rent observations.

### Possible improvements
- Use cross-validation and hyperparameter tuning.
- Develop a more sophisticated locality encoding strategy.
- Add richer geographic/property features.
- Try log transformation of Rent because of its right-skew.
- Use larger and more recent rental datasets.

## 10. Conclusion

The project demonstrates an end-to-end regression workflow for house-rent prediction. Four models were trained and compared using the required regression metrics. Random Forest was selected based on the strongest R² and lowest RMSE among the compared models, while Gradient Boosting produced the lowest MAE.

## 11. Demo

The project includes a Streamlit application. It accepts property characteristics and returns a predicted monthly rent using the saved Random Forest model.
