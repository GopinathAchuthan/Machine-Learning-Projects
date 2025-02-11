## Comprehensive Guide to Regression Model Evaluation Metrics
In regression tasks, several metrics are commonly used to evaluate the performance of a model. These metrics measure the difference between the predicted and actual values, helping assess how well the model fits the data. Here are the key regression model evaluation metrics:

### 1. **Mean Absolute Error (MAE)**
- **Definition**: MAE calculates the average of the absolute errors between the predicted and actual values.
- **Formula**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  
  Where:
  - $y_i$ is the actual value,
  - $\hat{y}_i$ is the predicted value,
  - $n$ is the total number of samples.
  
- **Interpretation**: MAE gives an average of the absolute differences between predictions and actual observations. It is easy to interpret and less sensitive to outliers compared to other metrics.

### 2. **Mean Squared Error (MSE)**
- **Definition**: MSE calculates the average of the squared errors, penalizing larger errors more than MAE.
- **Formula**:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Interpretation**: MSE is more sensitive to outliers due to the squaring of errors. It is commonly used in regression problems but can be misleading if the data contains outliers.

### 3. **Root Mean Squared Error (RMSE)**
- **Definition**: RMSE is the square root of the MSE, giving the error in the same units as the original data.
- **Formula**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Interpretation**: RMSE provides an error measure in the same units as the target variable, which makes it easier to interpret. Like MSE, RMSE is sensitive to large errors due to the squaring of the residuals.

### 4. **R-squared (R²)**
- **Definition**: R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Formula**:
  $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$


  Where:
  - $\bar{y}$ is the mean of the actual values.

- **Interpretation**: R² values range from 0 to 1, where 1 indicates perfect prediction and 0 indicates that the model does not explain any variance. A higher R² indicates a better fit of the model to the data, but it should be used carefully, as it may be misleading in certain situations.

### 5. **Adjusted R-squared (Adjusted R²)**
- **Definition**: Adjusted R² adjusts R² for the number of predictors in the model. It is useful when comparing models with a different number of features.
- **Formula**:
  $$\text{Adjusted } R^2 = 1 - \left(1 - R^2 \right) \frac{n - 1}{n - p - 1}$$

  Where:
  - $n$ is the number of data points,
  - $p$ is the number of features in the model.

- **Interpretation**: Adjusted R² is a better metric than R² when comparing models with different numbers of features. It penalizes the addition of unnecessary variables to the model.

### 6. **Mean Absolute Percentage Error (MAPE)**
- **Definition**: MAPE calculates the percentage difference between predicted and actual values, providing an intuitive understanding of how much error is present in percentage terms.
- **Formula**:
  $$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$$

- **Interpretation**: MAPE gives a percentage error, making it easy to interpret in business and real-world applications. However, it can be problematic when actual values are very small or zero, leading to infinite or undefined values.

### 7. **Explained Variance Score**
- **Definition**: This metric measures how much of the variance in the target variable is explained by the model.
- **Formula**:
  $$\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$

- **Interpretation**: A higher explained variance score indicates that the model has captured more of the variability in the target variable.

---

### **Summary of Metrics**:

| **Metric**              | **Formula**                                      | **Interpretation**                                                      |
|-------------------------|-------------------------------------------------|-------------------------------------------------------------------------|
| **MAE**                 | $$\frac{1}{n} \sum_{i} \|y_i - \hat{y}_i\|$$        | Average absolute error, less sensitive to outliers                        |
| **MSE**                 | $\frac{1}{n} \sum_{i} (y_i - \hat{y}_i)^2$     | Average squared error, more sensitive to large errors                   |
| **RMSE**                | $\sqrt{\frac{1}{n} \sum_{i} (y_i - \hat{y}_i)^2}$ | Root of MSE, interpretable in the same units as the target              |
| **R²**                  | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of variance explained by the model                           |
| **Adjusted R²**         | $1 - \left(1 - R^2 \right) \frac{n - 1}{n - p - 1}$ | R² adjusted for the number of predictors in the model                    |
| **MAPE**                | $\frac{1}{n} \sum_{i} \left\| \frac{y_i - \hat{y}_i}{y_i} \right\| \times 100$ | Percentage error, sensitive to small actual values                       |
| **Explained Variance**  | $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$ | Measures the proportion of variance explained by the model               |

### Choosing the Right Metric:
- Use **MAE** for simplicity when you care equally about all errors.
- **MSE** or **RMSE** is better when large errors are undesirable and you want to penalize them more.
- Use **R²** for a quick understanding of model performance, but consider **Adjusted R²** when comparing models with different numbers of predictors.
- **MAPE** is useful for business contexts where you need percentage errors, but be cautious with zero or small values.
- **Explained Variance** helps in evaluating how well the model explains the variance in the data.

Each of these metrics has its advantages and limitations, so it's important to choose the one that best fits your specific regression problem.
