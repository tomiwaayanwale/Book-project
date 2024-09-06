# Splendor Hotel Groups (SHG) Booking Data Analysis

## Project Overview

This project involves analyzing booking data for Splendor Hotel Groups (SHG) to provide insights into customer behavior, cancellation patterns, revenue trends, and operational efficiency. The goal is to leverage data to optimize business strategies, reduce cancellations, and improve revenue generation.

## Objectives

### 1. **Booking Patterns:**
- Identify trends in booking over time and seasons.
- Analyze how lead times vary by booking channel and customer type.

### 2. **Customer Behavior:**
- Determine which distribution channels contribute most to bookings.
- Analyze the impact of guests' country of origin on revenue.

### 3. **Cancellation Analysis:**
- Identify factors correlated with cancellations and predict potential cancellations.
- Compare revenue loss from cancellations across different segments.

### 4. **Revenue Optimization:**
- Explore overall revenue trends and the contribution of various customer segments or countries.
- Identify optimal pricing strategies based on the Average Daily Rate (ADR).

### 5. **Geographical Analysis:**
- Analyze the geographical distribution of guests and its impact on marketing efforts.
- Identify the relationship between country of origin and extended stays or cancellations.

### 6. **Operational Efficiency:**
- Analyze the average length of stay and its variation by booking channel.
- Determine staffing needs based on check-out patterns.

### 7. **Impact of Deposit Types:**
- Assess how deposits impact cancellations and revenue.
- Identify patterns in deposit usage across different customer segments.

### 8. **Corporate Bookings:**
- Analyze corporate bookings and compare their ADR to other customer types.
- Identify trends in corporate bookings that can inform business strategies.

### 9. **Time-to-Event Analysis:**
- Study how the lead time (time between booking and arrival) affects revenue and cancellations.
- Identify lead time ranges associated with higher satisfaction or revenue.

### 10. **Comparison of Online and Offline Travel Agents:**
- Compare the revenue contribution of online and offline travel agents.
- Study how cancellation rates vary between these booking types.

## Dataset

The dataset contains the following key features:
- `Booking Details`: Information about booking dates, arrival/departure, lead time, and booking channels.
- `Customer Details`: Information on the customer type, origin country, and loyalty programs.
- `Revenue Details`: Details about room rates (ADR), length of stay, and booking status (canceled or not).
- `Operational Features`: Check-out dates and resource usage data.

## Data Preprocessing

The following preprocessing steps were performed:
1. **Handling Missing Values**: Missing numeric values were imputed with the median, while categorical missing values were replaced with the most frequent values.
2. **Encoding Categorical Data**: One-hot encoding was applied to transform categorical variables into a format suitable for machine learning algorithms.
3. **Feature Scaling**: Applied to numeric columns to normalize the data where necessary.

## Machine Learning Models

The project employed a logistic regression model to predict cancellations. The model pipeline included:
1. Data preprocessing (imputation and encoding).
2. Fitting the logistic regression model.
3. Evaluating model performance using classification metrics (accuracy, precision, recall, F1-score).

### Code

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define the features (X) and target (y)
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define the transformers
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline
model = LogisticRegression(max_iter=1000)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Print evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

The logistic regression model achieved the following performance on the test set:
- **Accuracy**: 86%
- **Precision**: 87% for non-canceled bookings, 84% for canceled bookings.
- **Recall**: 91% for non-canceled bookings, 78% for canceled bookings.
- **F1-Score**: 89% for non-canceled bookings, 81% for canceled bookings.

## Conclusion

The analysis provides key insights into booking patterns, customer behavior, and operational efficiency. By implementing the suggested recommendations based on data-driven insights, SHG can improve overall revenue, optimize staffing, and reduce cancellation rates.

## Future Work

Further analysis could include:
- Analyzing the impact of promotions and loyalty programs on booking behavior.
- Developing advanced predictive models for customer lifetime value (CLV).
- Enhancing time-to-event analysis using survival models.

## Installation and Usage

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn plotly
   ```
2. Run the Jupyter notebook containing the analysis and code.

## Author

Lead Analyst: Tomiwa Oluwaferanmi
