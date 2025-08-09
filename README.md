Credit Card Default Prediction Analysis

This project focuses on analyzing a credit card dataset to build a predictive model for identifying customers who are likely to default on their credit card payments. The dataset contains various attributes about customers, including demographic information, credit behavior, transaction history, and their default status. The primary objective is to develop a machine learning model that can accurately predict credit card default, which can help financial institutions in risk assessment and targeted interventions.

Based on the initial analysis of the dataset:

*   **Number of Rows and Columns:** The dataset contains 10000 rows and 24 columns, as indicated by `data.shape`.

*   **Data Types:** The dataset includes features with the following data types, as seen in `data.info()`:
    *   `int64`: 14 columns (e.g., Age, Annual_Income, Credit_Score, Defaulted)
    *   `float64`: 5 columns (e.g., Credit_Utilization_Ratio, Debt_To_Income_Ratio)
    *   `object`: 5 columns (e.g., Customer_ID, Gender, Marital_Status, Education_Level, Employment_Status)

*   **Summary of Numerical Features:** The descriptive statistics (`data.describe()`) provide insights into the distribution of numerical features:
    *   `Age` ranges from 21 to 69 with a mean around 45.
    *   `Annual_Income` has a wide range, with a mean around 60267.
    *   `Credit_Score` ranges from 469 to 832, with a mean around 649.
    *   `Credit_Utilization_Ratio` and `Debt_To_Income_Ratio` are between 0 and 1.
    *   Other numerical features like `Number_of_Late_Payments`, `Total_Transactions_Last_Year`, `Total_Spend_Last_Year`, `CLV`, and various transaction-related amounts show varying ranges and distributions.

*   **Summary of Categorical Features:** The unique value counts and sample unique values for object type columns reveal the categories within these features:
    *   `Customer_ID`: Unique identifier for each customer (10000 unique values).
    *   `Gender`: Contains 'Male' and 'Female' categories (2 unique values).
    *   `Marital_Status`: Includes 'Married', 'Divorced', and 'Single' categories (3 unique values).
    *   `Education_Level`: Consists of 'PhD', 'High School', 'Bachelor', and 'Master' categories (4 unique values).
    *   `Employment_Status`: Contains 'Unemployed', 'Employed', and 'Self-employed' categories (3 unique values).

There are no missing values in the dataset based on the `data.isnull().sum()` output.

%%markdown
## Exploratory Data Analysis (EDA)

The exploratory data analysis provided valuable insights into the dataset and the relationships between features and the target variable, 'Defaulted'.

### Distribution of Numerical Features

Histograms and box plots of numerical features revealed various distributions. Some features, such as `Annual_Income` and `Credit_Utilization_Ratio`, showed evidence of outliers, as seen in the box plots. The distributions of other features like `Age`, `Credit_Score`, and `Tenure_in_Years` appeared more normally distributed or showed expected ranges.

### Distribution of Categorical Features and Target Variable

Count plots for categorical features illustrated the distribution of observations across different categories. For instance, the distribution of `Gender`, `Marital_Status`, `Education_Level`, and `Employment_Status` were visualized. The count plot for the target variable, `Defaulted`, indicated a class imbalance, with a higher number of customers who did not default (Defaulted = 0) compared to those who did (Defaulted = 1).

### Relationships Between Features

Pair plots of key numerical features (`Annual_Income`, `Credit_Score`, `Credit_Utilization_Ratio`, `Debt_To_Income_Ratio`) colored by the `Defaulted` status suggested some potential separation between the two classes based on `Credit_Score` and `Credit_Utilization_Ratio`. The heatmap of the correlation matrix for numerical features showed generally weak correlations between most pairs of features. Notable correlations were observed between `Total_Spend_Last_Year` and `Annual_Income`, and between some transaction-related features.

### Relationship Between Categorical Features and Target Variable

Analyzing the defaulted status by categorical features using count plots and normalized value counts showed that while there were variations in the counts within each category, the proportion of defaulted customers was relatively similar across the categories of `Gender`, `Marital_Status`, `Education_Level`, and `Employment_Status`. This suggests that these categorical features, individually, might not be strong predictors of default, although their interactions or combinations with other features could be relevant.

Overall, the EDA highlighted the need to address outliers in certain numerical features and acknowledged the class imbalance in the target variable, which will be important considerations for subsequent data preprocessing and model training steps.


%%markdown
## Data Cleaning and Preprocessing

Based on the insights gained from the Exploratory Data Analysis (EDA), the following steps were taken to clean and preprocess the data for machine learning:

### 1. Handling Outliers

Potential outliers were identified in numerical features such as `Annual_Income` and `Credit_Utilization_Ratio` through box plots during the EDA phase. To mitigate their impact on model performance, these outliers were handled by **capping** their values at the 99th percentile. This approach limits extreme values without removing the data points entirely.

### 2. Feature Engineering

New features were created to potentially enhance the predictive power of the model:
- `Income_Per_Transaction`: Calculated as `Annual_Income` divided by `Total_Transactions_Last_Year`. This feature aims to capture the income generated per transaction.
- `Avg_Monthly_Spend`: Calculated as `Total_Spend_Last_Year` divided by 12. This provides a measure of average monthly expenditure.
- `Late_Payment_Ratio`: Calculated as `Number_of_Late_Payments` divided by `Total_Transactions_Last_Year`. This represents the proportion of late payments relative to total transactions.
- `Credit_Score_x_Credit_Utilization`: An interaction term created by multiplying `Credit_Score` and `Credit_Utilization_Ratio`. This could capture a combined effect of creditworthiness and credit usage.
- `Age_Group`: Categorized `Age` into distinct bins (e.g., '20-29', '30-39'). This transforms a continuous variable into a categorical one, potentially revealing non-linear relationships with the target.
- `Income_Bracket`: Categorized `Annual_Income` into quantile-based bins (e.g., 'Very Low', 'Low', 'Medium', 'High', 'Very High'). This creates income groups with roughly equal numbers of customers.

Division by zero in the ratio calculations (`Income_Per_Transaction` and `Late_Payment_Ratio`) was handled by replacing zero values in the denominator with a small value or NaN and then filling resulting NaN values with the median of the respective new feature to avoid infinite values.

### 3. Encoding Categorical Variables

Categorical features (`Gender`, `Marital_Status`, `Education_Level`, `Employment_Status`, `Age_Group`, `Income_Bracket`) were converted into a numerical format suitable for machine learning models using **One-Hot Encoding**. This process creates new binary columns for each category within a feature, preventing the model from assuming any ordinal relationship between categories. The 'Customer_ID' column was excluded as it is a unique identifier and not relevant for modeling.

### 4. Scaling Numerical Features

Numerical features (including the newly engineered numerical features) were scaled using **Standard Scaling**. This technique standardizes the features by removing the mean and scaling to unit variance. Scaling is important for many machine learning algorithms that are sensitive to the scale of input features, such as Logistic Regression and algorithms that use distance metrics.

These cleaning and preprocessing steps were applied to prepare the data for the subsequent feature selection and model training phases, aiming to improve the performance and robustness of the predictive model.


%%markdown
## Feature Selection

Feature selection is a crucial step in the machine learning pipeline that involves choosing the most relevant features from the dataset to use for model training. This helps in reducing dimensionality, improving model performance, reducing training time, and enhancing the interpretability of the model.

In this project, we employed the **SelectKBest** method with the **f_classif** score function to select the top features.

**SelectKBest:** This method selects features based on the k highest scores according to a specified scoring function.
**f_classif:** This is a scoring function used for classification tasks. It computes the ANOVA F-value for the relationship between each feature and the target variable. Features with higher F-values are considered more statistically significant and thus more likely to be relevant for prediction.

We selected the top 15 features using this method. The selected features are:

*   Credit_Score
*   Credit_Utilization_Ratio
*   Number_of_Late_Payments
*   Total_Spend_Last_Year
*   Min_Transaction_Amount
*   Unique_Merchant_Categories
*   Avg_Monthly_Spend
*   Late_Payment_Ratio
*   Credit_Score_x_Credit_Utilization
*   Gender_Female
*   Gender_Male
*   Employment_Status_Employed
*   Employment_Status_Unemployed
*   Age_Group_40-49
*   Age_Group_60-69

These features were selected because they demonstrated the highest F-statistic scores with respect to the 'Defaulted' target variable, indicating a stronger potential relationship and predictive capability compared to the unselected features.


%%markdown
## Model Training and Evaluation

Several classification models were trained and evaluated to predict credit card default. The models chosen were Logistic Regression, Random Forest, and Gradient Boosting, which are commonly used and effective algorithms for binary classification tasks.

### Trained Models

1.  **Logistic Regression:** A linear model that uses a logistic function to model the probability of a binary outcome. It's a simple yet powerful baseline model.
2.  **Random Forest:** An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes of the individual trees. It is known for its robustness and ability to handle non-linear relationships.
3.  **Gradient Boosting:** Another ensemble learning method that builds trees sequentially, where each new tree attempts to correct the errors of the previous ones. It often provides high predictive accuracy.

Each of these models was trained on the training dataset (`X_train_split`, `y_train_split`) using their default parameters.

### Evaluation Metrics

The trained models were evaluated on the validation set (`X_val`, `y_val`) using several key metrics appropriate for binary classification, especially considering the potential class imbalance:

*   **Accuracy:** The proportion of correctly classified instances.
*   **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to a low false positive rate.
*   **Recall (Sensitivity):** The ratio of correctly predicted positive observations to the all observations in the actual class. High recall relates to a low false negative rate.
*   **F1-score:** The harmonic mean of Precision and Recall. It provides a single metric that balances both concerns.
*   **AUC (Area Under the ROC Curve):** Represents the model's ability to distinguish between the positive and negative classes. A higher AUC indicates better discriminatory power.

The evaluation results on the validation set were as follows:


## Model Evaluation Results on Validation Set:

**Logistic Regression:**
  Accuracy: 0.6725
  Precision: 0.5650
  Recall: 0.3298
  F1-score: 0.4165
  AUC: 0.6888

**Random Forest:**
  Accuracy: 0.6488
  Precision: 0.5074
  Recall: 0.3034
  F1-score: 0.3797
  AUC: 0.6604

**Gradient Boosting:**
  Accuracy: 0.6656
  Precision: 0.5494
  Recall: 0.3139
  F1-score: 0.3996
  AUC: 0.6781

### Model Selection

The best performing model was selected based on its performance on the validation set. Considering the nature of credit card default prediction, where correctly identifying potential defaulters (high recall) is important, alongside minimizing false positives (high precision), the **AUC score** was chosen as the primary metric for model selection. AUC provides a good overall measure of a model's ability to discriminate between the positive and negative classes, balancing the trade-off between sensitivity and specificity.

Based on the AUC scores from the validation set:

*   Logistic Regression: 0.6888
*   Random Forest: 0.6604
*   Gradient Boosting: 0.6781

The **Logistic Regression** model achieved the highest AUC score (0.6888) on the validation set. While its recall is not the highest, its balance across precision, recall, and overall discriminatory power (as indicated by AUC) made it the best choice among the three models for the final evaluation on the test set.


## Model Evaluation on Test Set

The selected Logistic Regression model was then evaluated on the unseen test set to obtain an unbiased estimate of its final performance. The evaluation results on the test set were as follows: Accuracy 0.6735, Precision 0.5318, Recall 0.3324, F1-score 0.4090, and AUC 0.6827.

## Conclusion and Future Work

The credit card default prediction analysis has been successfully conducted. A Logistic Regression model was trained and evaluated, showing moderate capability in predicting default. While its accuracy is reasonable, the relatively low Recall and F1-score indicate challenges in identifying all default cases. The class imbalance in the dataset likely contributed to these results.

As next steps, it is recommended to:
- Address class imbalance using techniques such as oversampling or undersampling.
- Perform hyperparameter tuning for the trained models to potentially improve their performance.
- Explore other classification models that might be more suitable.
- Develop more sophisticated features based on deeper domain understanding.

By undertaking these future steps, the credit card default prediction model can hopefully be improved to provide more accurate and useful results for risk-based decision making.
