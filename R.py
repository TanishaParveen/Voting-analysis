# Ridge Logistic Regression for Turnout Prediction (State-wise Analysis)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("synthetic_voting_dataset.csv")
original_states = df['State'].unique()  # Save original state names before encoding

# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'State', 'Constituency', 'Income_Level',
                      'Voted_Party', 'Turnout', 'Employment_Status', 'Education_Level']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

def run_ridge_analysis(state_name=None):
    """Run Ridge Logistic Regression analysis for a specific state or all states"""
    if state_name:
        # Get encoded state value
        state_encoded = label_encoders['State'].transform([state_name])[0]
        state_df = df[df['State'] == state_encoded]
        analysis_title = f"State: {state_name}"
    else:
        state_df = df
        analysis_title = "All States"
    
    # Prepare data for Turnout prediction
    X = state_df.drop(columns=['Turnout', 'Voter_ID', 'Date'])
    y = state_df['Turnout']
    
    # Check if we have enough samples and classes
    if len(y.unique()) < 2:
        print(f"\nCannot perform analysis for {analysis_title} - only one class present in Turnout")
        return None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ridge Logistic Regression (L2)
    ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Feature coefficients (L2 doesn't perform feature selection but we can still examine coefficients)
    feature_coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': ridge_model.coef_[0],
        'Absolute_Coefficient': np.abs(ridge_model.coef_[0])
    }).sort_values('Absolute_Coefficient', ascending=False)
    
    return accuracy, report, feature_coefficients

# Run analysis for all states first
print("=== Global Ridge Logistic Regression Analysis ===")
global_accuracy, global_report, global_coefficients = run_ridge_analysis()
if global_accuracy is not None:
    print(f"Accuracy: {global_accuracy * 100:.2f}%")
    print(global_report)
    
    # Save global report
    with open("ridge_logistic_regression_report.txt", "w") as f:
        f.write("=== Ridge Logistic Regression Report (All States) ===\n")
        f.write(f"Accuracy: {global_accuracy * 100:.2f}%\n\n")
        f.write(global_report)
        f.write("\n\nFeature Coefficients:\n")
        f.write(global_coefficients.to_string())
    
    # Plot feature coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(data=global_coefficients.head(15), x='Coefficient', y='Feature')
    plt.title("Top Feature Coefficients (Ridge Logistic Regression - All States)")
    plt.tight_layout()
    plt.savefig("ridge_feature_coefficients_all_states.png")
    plt.close()

# Run state-wise analysis
print("\n=== State-wise Ridge Logistic Regression Analysis ===")
state_results = []

for state in original_states:
    print(f"\nAnalyzing state: {state}")
    accuracy, report, coefficients = run_ridge_analysis(state)
    
    if accuracy is not None:
        state_results.append({
            'State': state,
            'Accuracy': accuracy,
            'Top_Feature': coefficients.iloc[0]['Feature'],
            'Top_Feature_Coefficient': coefficients.iloc[0]['Coefficient']
        })
        
        # Save state report
        with open(f"ridge_logistic_regression_{state}_report.txt", "w") as f:
            f.write(f"=== Ridge Logistic Regression Report ({state}) ===\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write(report)
            f.write("\n\nFeature Coefficients:\n")
            f.write(coefficients.to_string())
        
        # Plot state feature coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(data=coefficients.head(10), x='Coefficient', y='Feature')
        plt.title(f"Top Feature Coefficients (Ridge Logistic Regression - {state})")
        plt.tight_layout()
        plt.savefig(f"ridge_feature_coefficients_{state}.png")
        plt.close()

# Create state comparison report if we have results
if state_results:
    state_comparison = pd.DataFrame(state_results).sort_values('Accuracy', ascending=False)
    
    print("\n=== State Performance Comparison ===")
    print(state_comparison[['State', 'Accuracy']].to_string(index=False))
    
    # Save state comparison
    state_comparison.to_csv("ridge_state_performance_comparison.csv", index=False)
    
    # Plot state accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=state_comparison, x='State', y='Accuracy')
    plt.title("Ridge Logistic Regression Accuracy by State")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ridge_state_accuracy_comparison.png")
    plt.close()
    
    # Plot top features by state
    plt.figure(figsize=(12, 6))
    sns.barplot(data=state_comparison, x='State', y='Top_Feature_Coefficient', hue='Top_Feature')
    plt.title("Most Influential Feature by State")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("ridge_state_top_features.png")
    plt.close()

print("\nAnalysis complete. All reports and visualizations saved.")