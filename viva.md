# Viva Questions — House Rent Prediction

1. **What is Machine Learning?**
Machine Learning is a method where computers learn patterns from data to make predictions or decisions without being explicitly programmed for every case.

2. **What is your problem statement?**
Predict monthly house rent from property characteristics.

3. **What is your dataset?**
The supplied House Rent Dataset containing 4,746 records and 12 columns.

4. **How many records are present?**
4,746 records.

5. **What is the target variable?**
Rent.

6. **What are the input features?**
BHK, Size, Floor-derived features, Area Type, City, Furnishing Status, Tenant Preferred, Bathroom, and derived posting-date features.

7. **Is the problem classification, regression, or clustering?**
Regression because Rent is a continuous numerical value.

8. **Why these algorithms?**
They provide a progression from a simple linear baseline to tree-based and ensemble regression models.

9. **What is preprocessing?**
Preparing raw data so that it can be consistently used by ML algorithms, including cleaning, transformation, encoding and scaling.

10. **Why handle missing values?**
Most algorithms cannot directly use missing numerical values. Imputation provides a valid numerical value when necessary.

11. **What is encoding?**
Converting categorical values into numerical representations that ML algorithms can process.

12. **Why split training and testing data?**
To evaluate how well the trained model performs on unseen data.

13. **What is X_train?**
The input features used for model training.

14. **What is y_train?**
The target Rent values corresponding to X_train.

15. **What does fit() do?**
It trains the model using the training data.

16. **What does predict() do?**
It generates predicted target values for new/unseen input data.

17. **Which evaluation metrics did you use?**
MAE, MSE, RMSE and R².

18. **What does R² mean?**
It measures the proportion of variance in the target explained by the model; higher is generally better for the same evaluation setup.

19. **What are the limitations?**
Limited dataset scope, high-cardinality locality removed, omitted real-world factors, one train/test split, and sensitivity to unusual observations.

20. **How would you improve it?**
Use cross-validation, hyperparameter tuning, better locality representation, richer features, larger/current data, and potentially transform the skewed rent target.
