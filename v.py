import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_voting_dataset.csv")
    return df

df = load_data()

# Title and description
st.title("🗳️ AI-Powered Voting System Analysis Dashboard")
st.markdown("This dashboard shows insights from synthetic voting data using AI models.")

# Sidebar for state selection
st.sidebar.header("State Selection")
all_states = df['State'].unique()
selected_state = st.sidebar.selectbox("Select a State for Analysis", ['All States'] + list(all_states))

# Filter data based on state selection
if selected_state == 'All States':
    filtered_df = df
    state_title = "All States"
else:
    filtered_df = df[df['State'] == selected_state]
    state_title = selected_state

# Show raw data
if st.checkbox("Show raw dataset"):
    st.dataframe(filtered_df.head(50))

# Preprocessing function
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ['Gender', 'State', 'Constituency', 'Income_Level',
                          'Voted_Party', 'Turnout', 'Employment_Status', 'Education_Level']
    
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Age vs Turnout Visualization
st.subheader(f"Age Distribution by Voter Turnout - {state_title}")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(data=filtered_df, x='Age', hue='Turnout', multiple='stack', bins=20, ax=ax1)
st.pyplot(fig1)

# Correlation Heatmap
st.subheader(f"Feature Correlation Matrix - {state_title}")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(filtered_df.drop(columns=['Date']).corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# State-specific analysis
if selected_state != 'All States':
    st.subheader(f"State-Specific Model Analysis - {selected_state}")
    
    # Preprocess data
    processed_df, label_encoders = preprocess_data(filtered_df.copy())
    
    # Turnout Prediction
    st.markdown("### Turnout Prediction")
    X_turnout = processed_df.drop(columns=['Turnout', 'Voter_ID', 'Date'])
    y_turnout = processed_df['Turnout']
    
    if len(y_turnout.unique()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_turnout, y_turnout, test_size=0.2, random_state=42)
        
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        st.markdown(f"**Accuracy:** {accuracy:.2%}")
        st.text_area("Classification Report", report, height=200)
    else:
        st.warning("Not enough classes in Turnout data for this state to perform prediction")
    
    # Voted Party Prediction
    st.markdown("### Voted Party Prediction")
    X_party = processed_df.drop(columns=['Voted_Party', 'Voter_ID', 'Date'])
    y_party = processed_df['Voted_Party']
    
    if len(y_party.unique()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_party, y_party, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        st.markdown(f"**Accuracy:** {accuracy:.2%}")
        st.text_area("Classification Report", report, height=200)
        
        # Feature Importance
        st.markdown("### Feature Importance for Party Prediction")
        features = X_party.columns
        importances = rf.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax3)
        plt.title('Feature Importances for Party Prediction')
        st.pyplot(fig3)
    else:
        st.warning("Not enough classes in Voted Party data for this state to perform prediction")

# Global model reports (only shown when "All States" is selected)
if selected_state == 'All States':
    st.subheader("Global Model Performance")
    
    # Function to extract accuracy from report text
    def extract_accuracy(text):
        match = re.search(r'Accuracy: (\d+\.\d+)%', text)
        return float(match.group(1)) if match else None
    
    reports = {}
    
    # Load Logistic Regression report
    try:
        with open("voting_analysis_report.txt", "r") as f:
            logistic_report = f.read()
            logistic_acc = extract_accuracy(logistic_report)
            reports['Logistic Regression'] = (logistic_acc, logistic_report)
    except FileNotFoundError:
        st.warning("Logistic Regression report not found")
    
    # Load Ridge Regression report
    try:
        with open("ridge_logistic_regression_report.txt", "r") as f:
            ridge_report = f.read()
            ridge_acc = extract_accuracy(ridge_report)
            reports['Ridge Regression'] = (ridge_acc, ridge_report)
    except FileNotFoundError:
        st.warning("Ridge Regression report not found")
    
    # Load Lasso Regression report
    try:
        with open("lasso_logistic_regression_report.txt", "r") as f:
            lasso_report = f.read()
            lasso_acc = extract_accuracy(lasso_report)
            reports['Lasso Regression'] = (lasso_acc, lasso_report)
    except FileNotFoundError:
        st.warning("Lasso Regression report not found")
    
    # Display best model if we have reports
    if reports:
        best_model = max(reports.items(), key=lambda x: x[1][0])
        st.subheader("🏆 Best Performing Model")
        st.markdown(f"**Model:** {best_model[0]}")
        st.markdown(f"**Accuracy:** {best_model[1][0]:.2f}%")
        
        # Let user select and view all reports
        st.subheader("📄 Model Performance Reports")
        model_choice = st.selectbox("Select a model to view detailed report:", list(reports.keys()))
        st.text_area("Model Evaluation Report", reports[model_choice][1], height=400)
        
        try:
            st.subheader("📊 Feature Importance Comparison Across Models")
            st.image("feature_importance_comparison.png", 
                   caption="Comparison of Feature Importances (Logistic, Ridge, Lasso)", 
                   use_container_width=True)
        except FileNotFoundError:
            st.warning("Feature importance comparison image not found")