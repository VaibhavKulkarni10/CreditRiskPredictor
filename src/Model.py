import streamlit as st
import numpy as np
import pandas as pd
import sklearn

st.title('Risk Performance Dashboard')

tab1, tab2, tab3 = st.tabs(["Data Overview", "Global Performance", "Local Performance"])

dataset = pd.read_csv("heloc_dataset_v1.csv")
dataset['RiskPerformance'] = dataset['RiskPerformance'].map({'Bad': 0, 'Good': 1})
data_encoded = pd.get_dummies(dataset, columns=['MaxDelq2PublicRecLast12M', 'MaxDelqEver'])

dataset['MaxDelqEver'] = dataset['MaxDelqEver'].map({
        1: 'No Such Value',
        2: 'Derogatory Comment',
        3: '120+ Days Delinquent',
        4: '90 Days Delinquent',
        5: '60 Days Delinquent',
        6: '30 days Delinquent',
        7: 'Unknown Delinquency',
        8: 'Current and Never Delinquent',
        9: 'All Other',
        })

dataset['MaxDelq2PublicRecLast12M'] = dataset['MaxDelq2PublicRecLast12M'].map({
        0: 'Derogatory Comment',
        1: '120+ Days Delinquent',
        2: '90 Days Delinquent',
        3: '60 Days Delinquent',
        4: '30 Days Delinquent',
        5: 'Unknown Delinquency',
        6: 'Unknown Delinquency',
        7: 'Current and Never Delinquent',
        8: 'All Other',
        9: 'All Other',
        })

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_encoded[['ExternalRiskEstimate', 'MSinceOldestTradeOpen']] = scaler.fit_transform(data_encoded[['ExternalRiskEstimate', 'MSinceOldestTradeOpen']])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc



# Split data into training and testing sets
X = data_encoded.drop(columns=['RiskPerformance'])
y = data_encoded['RiskPerformance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3959656)

rmm = RandomForestClassifier()
rmm.fit(X_train, y_train)

y_rmm = rmm.predict(X_test)

accuracy = accuracy_score(y_test, y_rmm)
precision = precision_score(y_test, y_rmm)
recall = recall_score(y_test, y_rmm)
f1 = f1_score(y_test, y_rmm)

# Calculate feature importances
importances = rmm.feature_importances_
feature_names = X_train.columns

sorted_indices = (-importances).argsort()[:10] 

with tab1:
    st.markdown("Overview with Data Statistics & Visualiztion")
    st.subheader("Total Statistics")
    
    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Total Satisfactory Trades
    with col1:
        total_sat = dataset['NumSatisfactoryTrades'].sum()
        st.write("Total Satisfactory Trades:", total_sat)

    # Total Total Trades
    with col2:
        total_trades = dataset['NumTotalTrades'].sum()
        st.write("Total Total Trades:", total_trades)

    # Total Trades in Last 12 Months
    with col1:
        total_trades_12 = dataset['NumTradesOpeninLast12M'].sum()
        st.write("Total Trades in Last 12 Months:", total_trades_12)

    # Total Inquiries in Last 6 Months
    with col2:
        total_inq_6 = dataset['NumInqLast6M'].sum()
        st.write("Total Inquiries in Last 6 Months:", total_inq_6)

    # Create a two-column layout for the following plots
    col1, col2, col3 = st.columns(3)

    with col1:
    # Check if 'RiskPerformance' column exists in the dataset
        if 'RiskPerformance' in dataset.columns:
            # Count the occurrences of 0 and 1 in 'RiskPerformance'
            risk_counts = dataset['RiskPerformance'].value_counts()

            # Create a Streamlit app
            st.subheader('Bar Chart for Risk Performance')
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(risk_counts.index, risk_counts.values)
            ax.set_xlabel('Risk Performance')
            ax.set_ylabel('Count')
            ax.set_xticks(risk_counts.index)
            ax.set_xticklabels(['0', '1'])

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("The 'RiskPerformance' column does not exist in the dataset.")
    
    # Boxplot of External Risk Estimate by Risk Performance
    with col2:
        st.subheader('Risk Estimate by Risk Performance')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='RiskPerformance', y='ExternalRiskEstimate', data=dataset)
        plt.xlabel('Risk Performance')
        plt.ylabel('External Risk Estimate')
        st.pyplot(fig2)

    # Distribution of External Risk Estimate
    with col3:
        st.subheader('Distribution of External Risk Estimate')
        plt.figure(figsize=(8, 5))
        sns.histplot(dataset['ExternalRiskEstimate'], bins=20, kde=True)
        plt.xlabel('External Risk Estimate')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    with col1:
        st.subheader("Risk Performance Pie Chart")

        # Create a pie chart based on the distribution of "RiskPerformance"
        risk_counts = dataset['RiskPerformance'].value_counts()
        
        fig, ax = plt.subplots()
        ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the pie chart using st.pyplot()
        st.pyplot(fig)

    with col2:
        st.subheader("Max Delinquency in Last 12 Months")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Count the occurrences of each value
        max_delq_counts = dataset['MaxDelq2PublicRecLast12M'].value_counts()

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(max_delq_counts.index, max_delq_counts.values)
        plt.xlabel("Max Delinquency Category")
        plt.ylabel("Count")
        plt.xticks(rotation=90, ha="right")  # Rotate x-axis labels for better readability

        # Display the bar chart using st.pyplot()
        st.pyplot()

    with col3:
        st.subheader("Max Delinquency Ever")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Count the occurrences of each value
        max_delq_counts = dataset['MaxDelqEver'].value_counts()

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(max_delq_counts.index, max_delq_counts.values)
        plt.xlabel("Max Delinquency Category")
        plt.ylabel("Count")
        plt.xticks(rotation=90, ha="right")  # Rotate x-axis labels for better readability

        # Display the bar chart using st.pyplot()
        st.pyplot()

with tab2:
    st.markdown("Model Metrics and Performance, this can be used as an indicator for the business to check if there is decrease in performance of the model with time as the new data keeps on adding.")
    st.subheader("Global Explanations")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_rmm)
        st.write(f'Accuracy: {accuracy:.2f}')

    with col2:
        precision = precision_score(y_test, y_rmm)
        st.write(f'Precision: {precision:.2f}')

    with col3:
        recall = recall_score(y_test, y_rmm)
        st.write(f'Recall: {recall:.2f}')
    
    with col4:
        f1 = f1_score(y_test, y_rmm)
        st.write(f'F1-Score: {f1:.2f}')

    col1, col2, col3= st.columns(3)

    with col1:
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_rmm)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot(cmap='Blues', values_format='d')
        st.pyplot()

    with col2:
        def plot_learning_curve(estimator, X, y, cv, train_sizes):
            train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes)
        
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            return train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
    
        def plot_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
            plt.figure(figsize=(10, 6))
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.title("Learning Curve")
            st.pyplot()

        st.subheader("Generate Learning Curve")
        estimator = st.selectbox("Select a Machine Learning Model", [rmm])
        cv = st.slider("Number of Cross-Validation Folds", min_value=2, max_value=10)
        train_sizes = np.linspace(0.1, 1.0, 5)  # Adjust as needed

        if st.button("Generate Learning Curve"):
            train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = plot_learning_curve(estimator, X, y, cv, train_sizes)
            plot_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    with col3:
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, rmm.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Display ROC curve and AUC
        st.subheader("ROC Curve and AUC Analysis")
        st.write(f'AUC: {roc_auc:.2f}')
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0,    1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot()

    with col1:
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_rmm})
        df.sort_values(by='Predicted', ascending=False, inplace=True)
        df['CumulativeActual'] = df['Actual'].cumsum()
        df['CumulativePredicted'] = df['Predicted'].cumsum()
        df['CumulativeRandom'] = np.linspace(0, df['Actual'].sum(), len(df))

        st.subheader("Lift Chart")
        plt.figure()
        plt.plot(range(1, len(df) + 1), df['CumulativeActual'], label='Actual')
        plt.plot(range(1, len(df) + 1), df['CumulativePredicted'], label='Predicted')
        plt.plot(range(1, len(df) + 1), df['CumulativeRandom'], label='Random')
        plt.xlabel('Number of Observations')
        plt.ylabel('Cumulative Response')
        plt.legend()
        st.pyplot()

    with col3:
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_rmm})
        df.sort_values(by='Predicted', ascending=False, inplace=True)
        df['CumulativeActual'] = df['Actual'].cumsum()
        df['CumulativePredicted'] = df['Predicted'].cumsum()
        df['CumulativeRandom'] = np.linspace(0, df['Actual'].sum(), len(df))

        st.subheader("Gain Chart")
        plt.figure()
        plt.plot(range(1, len(df) + 1), df['CumulativeActual'] / df['Actual'].sum(), label='Actual')
        plt.plot(range(1, len(df) + 1), df['CumulativePredicted'] / df['Actual'].sum(), label='Predicted')
        plt.plot(range(1, len(df) + 1), df['CumulativeRandom'] / df['Actual'].sum(), label='Random')
        plt.xlabel('Number of Observations')
        plt.ylabel('Cumulative Gain')
        plt.legend()
        st.pyplot()

with tab3:
    st.markdown("Local Explanations where the model showcases the model confidence/probability by changing the slider along with a graph showing the changing in pattern of the features too using Permutation Importance.")
    st.subheader("Local Model Explanations")

    # Sidebar for user interaction
    st.sidebar.header("Input Features")

    # Create sliders for input features
    sliders = {}
    for ingredient in X_test.columns:
        min_value = float(X_test[ingredient].min())
        max_value = float(X_test[ingredient].max())
        default_value = (min_value + max_value) / 2
        sliders[ingredient] = st.sidebar.slider(label=ingredient, min_value=min_value, max_value=max_value, value=default_value)

    # Make predictions with the model
    prediction = rmm.predict([list(sliders.values())])
    predicted_class = prediction[0]

    # Display model prediction
    st.subheader("Model Prediction")
    st.write(f"Predicted Class: {predicted_class}")

    # Calculate and display model confidence (probability)
    probs = rmm.predict_proba([list(sliders.values())])
    probability = probs[0][predicted_class]
    st.subheader("Model Confidence")
    st.write(f"Probability: {probability:.2%}")

    # Calculate permutation feature importances
    perm_importance = permutation_importance(rmm, X_test, y_test, n_repeats=10, random_state=0)

    # Plot the feature importances
    st.subheader("Feature Importances")
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(10, 10))
    plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel('Permutation Importance')
    st.pyplot(plt)