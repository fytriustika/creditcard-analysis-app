# credit_card_user_dashboard.py

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

st.set_page_config(page_title='Credit Card User Analysis Dashboard', layout='wide')

@st.cache_data
def load_data():
    # Upload the file on Streamlit or set the path accordingly
    # data = pd.read_csv('/content/drive/MyDrive/Credit_Card_Dataset.csv')
    uploaded_file = st.sidebar.file_uploader("Upload Credit Card Dataset CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("Please upload a CSV file to continue.")
        return None

def initial_eda(data):
    st.header("Initial Data Exploration")
    st.write("**Shape of Data (rows, columns):**", data.shape)
    
    st.write("**First 5 Rows:**")
    st.dataframe(data.head())
    
    st.write("**Column Data Types:**")
    st.dataframe(pd.DataFrame(data.dtypes, columns=["Type"]))
    
    st.write("**Missing Values per Column:**")
    st.dataframe(data.isnull().sum())
    
    st.write("**Unique Values per Column:**")
    st.dataframe(pd.DataFrame({col: data[col].nunique() for col in data.columns}, index=["Unique Values"]).T)
    
    st.write("**Descriptive Statistics (Numerical):**")
    st.dataframe(data.describe())

def plot_distributions(data):
    st.subheader("Numerical Feature Distributions")
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if "Defaulted" in numerical_cols:
        numerical_cols = numerical_cols.drop("Defaulted")
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3))
        sns.histplot(data[col], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {col}")
        sns.boxplot(x=data[col], ax=ax2)
        ax2.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

def plot_categorical(data):
    st.subheader("Categorical Feature Distributions")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(x=data[col], ax=ax)
        ax.set_title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    if "Defaulted" in data.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=data['Defaulted'], ax=ax)
        ax.set_title("Distribution of Defaulted")
        st.pyplot(fig)

def plot_pairplot(data):
    st.subheader("Pairplot & Correlation Heatmap")
    key_numerical_cols = ['Annual_Income', 'Credit_Score', 'Credit_Utilization_Ratio', 'Debt_To_Income_Ratio', 'Defaulted']
    available_cols = [col for col in key_numerical_cols if col in data.columns]
    if len(available_cols) > 1:
        st.write("Pairplot (this might take time for large datasets):")
        fig = sns.pairplot(data[available_cols], hue='Defaulted' if 'Defaulted' in data.columns else None)
        st.pyplot(fig)
        st.write("Correlation Heatmap:")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(data[available_cols].corr(), annot=True, ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Not enough columns for pairplot/correlation heatmap.")

def handle_outliers(data):
    st.subheader("Handle Outliers")
    modified_data = data.copy()
    for col in ['Annual_Income', 'Credit_Utilization_Ratio']:
        if col in modified_data.columns:
            upper_bound = modified_data[col].quantile(0.99)
            modified_data[col] = modified_data[col].clip(upper=upper_bound)
    st.write("Outliers in 'Annual_Income' and 'Credit_Utilization_Ratio' are capped at the 99th percentile.")
    return modified_data

def feature_engineering(data):
    st.subheader("Feature Engineering")
    df = data.copy()
    # Income_Per_Transaction
    if 'Annual_Income' in df.columns and 'Total_Transactions_Last_Year' in df.columns:
        df['Income_Per_Transaction'] = df['Annual_Income'] / df['Total_Transactions_Last_Year'].replace(0, np.nan)
        df['Income_Per_Transaction'].fillna(df['Income_Per_Transaction'].median(), inplace=True)
    # Avg_Monthly_Spend
    if 'Total_Spend_Last_Year' in df.columns:
        df['Avg_Monthly_Spend'] = df['Total_Spend_Last_Year'] / 12
    # Late_Payment_Ratio
    if 'Number_of_Late_Payments' in df.columns and 'Total_Transactions_Last_Year' in df.columns:
        df['Late_Payment_Ratio'] = df['Number_of_Late_Payments'] / df['Total_Transactions_Last_Year'].replace(0, np.nan)
        df['Late_Payment_Ratio'].fillna(df['Late_Payment_Ratio'].median(), inplace=True)
    # Credit_Score x Credit_Utilization
    if 'Credit_Score' in df.columns and 'Credit_Utilization_Ratio' in df.columns:
        df['Credit_Score_x_Credit_Utilization'] = df['Credit_Score'] * df['Credit_Utilization_Ratio']
    # Age_Group
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70], labels=['20-29', '30-39', '40-49', '50-59', '60-69'], right=False)
    # Income_Bracket
    if 'Annual_Income' in df.columns:
        df['Income_Bracket'] = pd.qcut(df['Annual_Income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    st.write("Feature engineering completed. New features added.")
    return df

def preprocess_data(data):
    st.subheader("Preprocessing (Scaling & Encoding)")
    df = data.copy()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Customer_ID' in categorical_features:
        categorical_features.remove('Customer_ID')
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Defaulted' in numerical_features:
        numerical_features.remove('Defaulted')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    pipeline = Pipeline([('preprocessor', preprocessor)])
    data_preprocessed_array = pipeline.fit_transform(df)
    onehot_columns = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_preprocessed_columns = numerical_features + list(onehot_columns)
    data_preprocessed = pd.DataFrame(data_preprocessed_array, columns=all_preprocessed_columns, index=df.index)
    data_preprocessed['Defaulted'] = df['Defaulted']
    st.write("Data has been scaled and encoded.")
    return data_preprocessed, all_preprocessed_columns

def select_features(data_preprocessed, all_preprocessed_columns, k=15):
    st.subheader(f"Feature Selection (Top {k} Features)")
    X = data_preprocessed.drop('Defaulted', axis=1)
    y = data_preprocessed['Defaulted']
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    st.write("Selected Features:", selected_features)
    return X[selected_features], y, selected_features

def split_data(X, y):
    st.subheader("Train/Validation/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    st.write(f"Train shape: {X_train_split.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train_split, X_val, X_test, y_train_split, y_val, y_test

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    st.subheader("Model Training & Validation Results")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "model": model,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "F1": f1_score(y_val, y_pred),
            "AUC": roc_auc_score(y_val, y_proba),
        }
    results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
    st.dataframe(results_df.style.format("{:.4f}"))
    best_model_name = results_df['AUC'].idxmax()
    st.success(f"Best performing model on validation set: {best_model_name}")
    return models[best_model_name]

def evaluate_on_test(best_model, X_test, y_test):
    st.subheader("Best Model Performance on Test Set")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    st.write({
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
    })

def main():
    st.title("Credit Card User Analysis & Default Prediction Dashboard")
    data = load_data()
    if data is None:
        st.stop()
    st.sidebar.markdown("## Navigation")
    options = [
        "Initial EDA", "Distributions", "Pairplot & Correlations",
        "Handle Outliers", "Feature Engineering", "Preprocessing & Feature Selection",
        "Model Training & Evaluation"
    ]
    choice = st.sidebar.radio("Select View", options)
    if choice == "Initial EDA":
        initial_eda(data)
    elif choice == "Distributions":
        plot_distributions(data)
        plot_categorical(data)
    elif choice == "Pairplot & Correlations":
        plot_pairplot(data)
    elif choice == "Handle Outliers":
        data_mod = handle_outliers(data)
        st.dataframe(data_mod.head())
    elif choice == "Feature Engineering":
        data_feat = feature_engineering(data)
        st.dataframe(data_feat.head())
    elif choice == "Preprocessing & Feature Selection":
        data_mod = handle_outliers(data)
        data_feat = feature_engineering(data_mod)
        data_preprocessed, all_preprocessed_columns = preprocess_data(data_feat)
        X_selected, y, selected_features = select_features(data_preprocessed, all_preprocessed_columns)
        st.write("Preview of Selected Feature Data:")
        st.dataframe(X_selected.head())
    elif choice == "Model Training & Evaluation":
        data_mod = handle_outliers(data)
        data_feat = feature_engineering(data_mod)
        data_preprocessed, all_preprocessed_columns = preprocess_data(data_feat)
        X_selected, y, selected_features = select_features(data_preprocessed, all_preprocessed_columns)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_selected, y)
        best_model = train_and_evaluate_models(X_train, y_train, X_val, y_val)
        evaluate_on_test(best_model, X_test, y_test)

if __name__ == "__main__":
    main()