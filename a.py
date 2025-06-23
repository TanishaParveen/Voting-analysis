# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("synthetic_voting_dataset.csv")

# Step 2: Preprocessing
# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'State', 'Constituency', 'Income_Level',
                       'Voted_Party', 'Turnout', 'Employment_Status', 'Education_Level']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Get list of available states (original names before encoding)
original_states = pd.read_csv("synthetic_voting_dataset.csv")['State'].unique()

# Function to get state analysis
def analyze_state(state_name):
    try:
        # Get the encoded value for the state
        state_encoded = label_encoders['State'].transform([state_name])[0]
        
        # Filter data for the selected state
        state_df = df[df['State'] == state_encoded]
        
        if len(state_df) == 0:
            print(f"No data available for state: {state_name}")
            return
        
        print(f"\n=== Analysis for State: {state_name} ===")
        
        # Step 3: State-specific EDA
        plt.figure(figsize=(10, 6))
        sns.histplot(data=state_df, x='Age', hue='Turnout', multiple='stack', bins=20)
        plt.title(f"Age Distribution by Turnout - {state_name}")
        plt.xlabel("Age")
        plt.ylabel("Number of Voters")
        plt.tight_layout()
        plt.savefig(f"{state_name}_age_turnout_distribution.png")
        plt.close()
        
        # Step 4: State-specific Turnout Prediction
        X_state = state_df.drop(columns=['Turnout', 'Voter_ID', 'Date'])
        y_state = state_df['Turnout']
        
        if len(y_state.unique()) < 2:
            print(f"\nCannot perform Turnout prediction for {state_name} - not enough classes in target variable")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_state, y_state, test_size=0.2, random_state=42)
            
            # Logistic Regression Model
            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            y_pred_log = logreg.predict(X_test)
            
            print(f"\nLogistic Regression Report for {state_name} (Turnout Prediction):")
            print(classification_report(y_test, y_pred_log))
            
            # Save turnout accuracy
            turnout_accuracy = accuracy_score(y_test, y_pred_log)
        
        # Step 5: State-specific Voted Party Prediction
        X2_state = state_df.drop(columns=['Voted_Party', 'Voter_ID', 'Date'])
        y2_state = state_df['Voted_Party']
        
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2_state, y2_state, test_size=0.2, random_state=42)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X2_train, y2_train)
        y2_pred = rf_model.predict(X2_test)
        
        print(f"\nRandom Forest Report for {state_name} (Voted Party Prediction):")
        print(classification_report(y2_test, y2_pred))
        
        # Save party prediction accuracy
        party_accuracy = accuracy_score(y2_test, y2_pred)
        
        # Step 6: Save State-specific Deliverables
        with open(f"{state_name}_voting_analysis_report.txt", "w") as report_file:
            report_file.write(f"=== Voting System AI Analysis for {state_name} ===\n\n")
            
            if 'turnout_accuracy' in locals():
                report_file.write("1. Logistic Regression (Turnout Prediction) - Accuracy: {:.2f}%\n".format(turnout_accuracy * 100))
                report_file.write("\nLogistic Regression Classification Report:\n")
                report_file.write(classification_report(y_test, y_pred_log))
            else:
                report_file.write("1. Turnout Prediction not performed - insufficient class distribution\n")
            
            report_file.write("\n2. Random Forest (Voted Party Prediction) - Accuracy: {:.2f}%\n".format(party_accuracy * 100))
            report_file.write("\nRandom Forest Classification Report:\n")
            report_file.write(classification_report(y2_test, y2_pred))
            
            report_file.write("\n\n=== Visuals ===\n")
            report_file.write(f"1. Age vs Turnout Distribution for {state_name} - saved as '{state_name}_age_turnout_distribution.png'\n")
        
        print(f"\nState analysis complete. Results saved to '{state_name}_voting_analysis_report.txt'")
        
    except Exception as e:
        print(f"Error analyzing state {state_name}: {str(e)}")

# Main program
print("Available states for analysis:")
for i, state in enumerate(original_states, 1):
    print(f"{i}. {state}")

while True:
    try:
        selection = input("\nEnter the number of the state you want to analyze (or 'q' to quit): ")
        if selection.lower() == 'q':
            break
            
        state_num = int(selection) - 1
        if 0 <= state_num < len(original_states):
            selected_state = original_states[state_num]
            analyze_state(selected_state)
        else:
            print("Invalid selection. Please enter a number from the list.")
    except ValueError:
        print("Please enter a valid number or 'q' to quit.")

print("\nState-wise analysis completed.") 