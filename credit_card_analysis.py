import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Credit Card User Analysis Dashboard")

# READ DATA FROM GITHUB
CSV_URL = "https://raw.githubusercontent.com/fytriustika/creditcard-analysis-app/main/Credit_Card_Dataset.csv"
data = pd.read_csv(CSV_URL)

st.subheader("Data Preview")
st.dataframe(data.head())
st.write("Jumlah data (baris, kolom):", data.shape)
st.write("Statistik deskriptif untuk data numerik:")
st.dataframe(data.describe())

# EDA: Distributions
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop(['Defaulted'])

st.subheader("Distribusi dan Box Plot Numerik")
fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(10, 4*len(numerical_cols)))
for i, col in enumerate(numerical_cols):
    sns.histplot(data[col], kde=True, ax=axes[i,0])
    axes[i,0].set_title(f'Distribution of {col}')
    sns.boxplot(x=data[col], ax=axes[i,1])
    axes[i,1].set_title(f'Box plot of {col}')
st.pyplot(fig)

categorical_cols = data.select_dtypes(include=['object']).columns
st.subheader("Distribusi Kategori")
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(5*len(categorical_cols), 5))
if len(categorical_cols) == 1:
    axes = [axes]
for i, col in enumerate(categorical_cols):
    sns.countplot(x=data[col], ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Outlier capping
for col in ['Annual_Income', 'Credit_Utilization_Ratio']:
    upper_bound = data[col].quantile(0.99)
    data[col] = data[col].clip(upper=upper_bound)

# Feature Engineering
data['Income_Per_Transaction'] = data['Annual_Income'] / data['Total_Transactions_Last_Year'].replace(0, np.nan)
data['Income_Per_Transaction'].fillna(data['Income_Per_Transaction'].median(), inplace=True)
data['Avg_Monthly_Spend'] = data['Total_Spend_Last_Year'] / 12
data['Late_Payment_Ratio'] = data['Number_of_Late_Payments'] / data['Total_Transactions_Last_Year'].replace(0, np.nan)
data['Late_Payment_Ratio'].fillna(data['Late_Payment_Ratio'].median(), inplace=True)
data['Credit_Score_x_Credit_Utilization'] = data['Credit_Score'] * data['Credit_Utilization_Ratio']
data['Age_Group'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70],
                           labels=['20-29', '30-39', '40-49', '50-59', '60-69'], right=False)
data['Income_Bracket'] = pd.qcut(data['Annual_Income'], q=5,
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Preprocessing
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Customer_ID' in categorical_features:
    categorical_features.remove('Customer_ID')
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Defaulted' in numerical_features:
    numerical_features.remove('Defaulted')

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
data_preprocessed_array = preprocessing_pipeline.fit_transform(data)
onehot_columns = preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_preprocessed_columns = numerical_features + list(onehot_columns)
data_preprocessed = pd.DataFrame(data_preprocessed_array, columns=all_preprocessed_columns, index=data.index)
data_preprocessed['Defaulted'] = data['Defaulted']

# Feature Selection
X = data_preprocessed.drop('Defaulted', axis=1)
y = data_preprocessed['Defaulted']
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices].tolist()
X_selected_df = X[selected_features]

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.2, random_state=42)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# MODEL TRAINING
st.subheader("Model Training and Evaluation")
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_split, y_train_split)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(y_val, y_val_pred),
        "Recall": recall_score(y_val, y_val_pred),
        "F1-score": f1_score(y_val, y_val_pred),
        "AUC": roc_auc_score(y_val, y_val_proba)
    }

st.write(pd.DataFrame(results).T)

best_model_name = max(results, key=lambda m: results[m]['AUC'])
st.success(f"Best model on validation set: {best_model_name}")

# Test the best model
best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
st.subheader("Best Model Test Set Performance")
st.write({
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred),
    "F1-score": f1_score(y_test, y_test_pred),
    "AUC": roc_auc_score(y_test, y_test_proba)
})

st.info("""
**Tips:**
- To deploy: Push this script and your CSV to your GitHub repo, then launch on [streamlit.io/cloud](https://streamlit.io/cloud) and select this file.
- All data reading is done from GitHub, so you don't need Google Drive or Colab.
""")