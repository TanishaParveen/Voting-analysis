# Voting-analysis
<br>
This project leverages AI-driven regression and probability distribution methods to predict voter turnout using a custom synthetic dataset. The dataset includes key attributes such as Voter_ID, Age, Gender, State, Constituency, Income_Level, Voted_Party, Turnout, Employment_Status, and Education_Level, which were preprocessed using encoding and normalization. The Turnout column (binary: Yes/No) serves as the target variable, with Logistic Regression forming the baseline model to estimate participation likelihood.
<br>
State-wise analysis revealed critical geographic disparities in voter behavior—some states exhibited strong turnout correlations with age or education, while others were more influenced by income or employment status. To capture these variations, we incorporated state-specific feature engineering, including historical turnout averages and political leaning categories. Geospatial visualizations (e.g., interactive choropleth maps) highlighted regional trends, while state-stratified models improved localized predictions.
<br>
To enhance model robustness, we implemented regularization techniques (Ridge & Lasso Regression), which helped reduce overfitting and improve generalization. Lasso regression identified the most impactful features (e.g., Age and Education had higher coefficients than Income in certain states**), while Ridge regression stabilized multicollinear predictors like Employment_Status and Income. These adjustments refined accuracy to ~60-70% while maintaining interpretability.
<br>
All insights are integrated into a dynamic Streamlit dashboard, enabling users to:
<br>
•	Filter turnout trends by state, age group, or income bracket,
<br>
•	Compare model performance (Logistic vs. Ridge/Lasso),
,br>
•	Visualize feature importance and probability distributions.
<br>
