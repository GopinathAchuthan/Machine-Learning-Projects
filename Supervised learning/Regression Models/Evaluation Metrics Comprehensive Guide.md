## Comprehensive Guide to Regression Model Evaluation Metrics

### 1. **Mean Absolute Error (MAE)**

- **Definition**: The average of the absolute differences between predicted and actual values.
- **Formula**:  
  $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
  
- **Interpretation**: Lower MAE indicates better model performance, showing the average error in the same units as the target variable.
- **Use**: Best for understanding the average magnitude of errors.

---

### 2. **Mean Squared Error (MSE)**

- **Definition**: The average of the squared differences between predicted and actual values.
- **Formula**:  
  $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
  
- **Interpretation**: Lower MSE indicates a better model. MSE penalizes larger errors more than smaller ones due to squaring the differences.
- **Use**: Best for situations where larger errors are more undesirable.

---

### 3. **Root Mean Squared Error (RMSE)**

- **Definition**: The square root of the Mean Squared Error (MSE), returning the error metric in the same units as the target variable.
- **Formula**:  
  $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
  
- **Interpretation**: Like MSE but more interpretable because it’s in the same units as the dependent variable. A lower RMSE means a better fit.
- **Use**: Commonly used when large errors are heavily penalized.

---

### 4. **R-squared (R²)**

- **Definition**: The proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Formula**:  
  $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
  
- **Interpretation**: R² ranges from 0 to 1. A value closer to 1 indicates a better fit, meaning the model explains a large portion of the variance.
- **Use**: Helps understand how well the model fits the data.

---

### 5. **Adjusted R-squared (Adjusted R²)**

- **Definition**: An adjustment of R² that accounts for the number of predictors in the model, penalizing models with unnecessary predictors.
- **Formula**:  
  $R^2_{\text{adjusted}} = 1 - \left(1 - R^2\right)\frac{n-1}{n-p-1}$
  
- **Interpretation**: A more reliable metric than R² when comparing models with different numbers of predictors. It penalizes the addition of non-significant predictors.
- **Use**: Best for comparing models with different numbers of predictors.

---

### 6. **Mean Absolute Percentage Error (MAPE)**

- **Definition**: The average of the absolute percentage differences between predicted and actual values.
- **Formula**:  
  $\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$
  
- **Interpretation**: Expressed as a percentage, MAPE provides a relative measure of the error, making it easier to understand across different scales.
- **Use**: Useful for understanding the relative prediction accuracy.
- **Note**: There is some variation in MAPE.
---

### 7. **Explained Variance Score**

- **Definition**: Measures the proportion of the variance in the target variable that is explained by the model.
- **Formula**:  
  $\text{Explained Variance Score} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$
  
- **Interpretation**: Values closer to 1 indicate that the model explains most of the variance.
- **Use**: Provides a simple measure of the proportion of variance explained by the model.

---

### 8. **F-statistic (for regression models)**

- **Definition**: A test statistic used to assess the overall significance of a regression model, comparing it to a model with no predictors.
- **Interpretation**: A higher F-statistic suggests that the model is a better fit than a model with no predictors.
- **Use**: Useful in hypothesis testing to determine if at least one predictor variable is significantly related to the outcome.

---

### 9. **Residual Plot (Diagnostic)**

- **Definition**: A visual tool used to plot residuals (errors) against predicted values to assess homoscedasticity, linearity, and the presence of outliers.
- **Interpretation**: Random scatter of residuals indicates a good fit. Patterns or non-randomness suggest potential issues with the model (e.g., non-linearity, heteroscedasticity).
- **Use**: Essential for diagnosing model assumptions and detecting issues such as outliers, non-linearity, and incorrect model assumptions.

---

### **Summary of Essential Metrics**

| Metric                          | Purpose                                                 | Key Insight                                           |
|----------------------------------|---------------------------------------------------------|-------------------------------------------------------|
| **MAE (Mean Absolute Error)**    | Measures average absolute error                        | Provides a direct measure of average prediction error |
| **MSE (Mean Squared Error)**     | Measures squared error, penalizing large errors more   | Sensitive to large errors                             |
| **RMSE (Root Mean Squared Error)** | Square root of MSE for interpretability               | Reflects error in same units as target variable       |
| **R² (R-squared)**               | Proportion of variance explained by the model          | Measures goodness of fit                             |
| **Adjusted R²**                  | Adjusted version of R² considering number of predictors | More reliable for comparing models with different predictors |
| **MAPE (Mean Absolute Percentage Error)** | Average percentage error                           | Relative error, useful for cross-model comparisons   |
| **Explained Variance Score**     | Proportion of variance explained by the model          | Indicates model’s explanatory power                  |
| **F-statistic**                  | Assesses model significance relative to no predictors  | Helps test if the model is meaningful                 |
| **Residual Plot**                | Visual tool to check model assumptions                 | Used to detect patterns, outliers, and assumptions violations |

---

### **Conclusion**

The **essential regression metrics** give you a comprehensive picture of model performance, both numerically and visually. Using a combination of these metrics helps you assess the accuracy, goodness-of-fit, and the validity of the model's assumptions.
