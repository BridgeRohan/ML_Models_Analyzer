import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto" # Use "auto" to respect system theme
)

# --- App Title and Description (New Style) ---
st.title("🤖 ML Model Analyzer for Teen Phone Addiction")
st.markdown("""
    Welcome to the interactive Machine Learning model comparator. 
    This tool allows you to train, evaluate, and compare different classification and clustering models on the Teen Phone Addiction dataset.
    **Select models from the sidebar to begin!**
""")
st.divider()

# --- Data Loading and Caching (No Changes) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data directly from the file path."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.drop(['ID', 'Name'], axis=1, errors='ignore')
    
    # Bin the Target Variable into 5 Categories based on the 1-10 scale
    bin_edges = [0, 2, 4, 6, 8, 10.1]
    labels = [0, 1, 2, 3, 4]
    class_names_str = ["Very Low", "Low", "Medium", "High", "Very High"]
    
    df['addiction_category'] = pd.cut(df['Addiction_Level'], bins=bin_edges, labels=labels, right=True, include_lowest=True)
    df.dropna(subset=['addiction_category'], inplace=True)
    df['addiction_category'] = df['addiction_category'].astype(int)

    X = df.drop(['Addiction_Level', 'addiction_category'], axis=1)
    y = df['addiction_category'].values

    categorical_features = ['Gender', 'Location', 'School_Grade', 'Phone_Usage_Purpose']
    numerical_features = [col for col in df.columns if col not in categorical_features and col not in ['Addiction_Level', 'addiction_category']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor, class_names_str, X

# --- Sidebar for Model Selection (No Changes) ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_options = [
        "Logistic Regression", "Decision Tree", "k-Nearest Neighbors (kNN)",
        "Support Vector Machine (SVM)", "Random Forest", "Neural Networks (NNs)", "K-means"
    ]
    selected_models = st.multiselect(
        "Choose ML models to compare",
        options=model_options,
        default=["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
    )
    st.info("The final summary table will appear at the bottom of the page after the models run.")


# --- Main App Body ---
if selected_models:
    # Load data directly from the local file
    file_name = 'teen_phone_addiction_dataset.csv'
    (X_train, X_test, y_train, y_test), preprocessor, class_names, X_full = load_and_preprocess_data(file_name)
    
    results_list = []
    st.header("📊 Model Performance Results")

    # --- Supervised Models Evaluation ---
    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-Nearest Neighbors (kNN)": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC(random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Neural Networks (NNs)": MLPClassifier(max_iter=1000, random_state=42)
    }
    
    for name in selected_models:
        if name in supervised_models:
            with st.expander(f"**{name}** Results", expanded=True):
                model = supervised_models[name]
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                
                # --- Training & Prediction ---
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = pipeline.predict(X_test)
                prediction_time = time.time() - start_time

                # --- Metrics Calculation ---
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                y_prob = pipeline.predict_proba(X_test)
                auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                
                model_filename = f'temp_{name}.joblib'
                joblib.dump(pipeline, model_filename)
                memory_kb = os.path.getsize(model_filename) / 1024
                os.remove(model_filename)
                
                eval_metrics_str = (f"Precision: {report['weighted avg']['precision']:.2f}, Recall: {report['weighted avg']['recall']:.2f}, "
                                    f"F-score: {report['weighted avg']['f1-score']:.2f}, AUC-ROC: {auc_roc:.2f}")
                
                # --- Display Results in Columns (New Style) ---
                col1, col2 = st.columns((1, 1))

                with col1:
                    st.subheader("📈 Performance Metrics")
                    metric1, metric2 = st.columns(2)
                    metric1.metric(label="Weighted F1-Score", value=f"{report['weighted avg']['f1-score']:.2f}")
                    metric2.metric(label="AUC-ROC Score", value=f"{auc_roc:.2f}")
                    st.markdown(f"**Accuracy:** {report['accuracy']:.2%}")
                    st.markdown(f"**Weighted Precision:** {report['weighted avg']['precision']:.2f}")
                    st.markdown(f"**Weighted Recall:** {report['weighted avg']['recall']:.2f}")
                    
                    st.divider()

                    st.subheader("⚙️ Computational Cost")
                    st.markdown(f"**⏱️ Training Time:** `{training_time:.4f} seconds`")
                    st.markdown(f"**⚡ Prediction Speed:** `{prediction_time:.4f} seconds`")
                    st.markdown(f"**💾 Memory Usage:** `{memory_kb:.2f} KB`")

                with col2:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_xlabel('Predicted Labels')
                    ax.set_ylabel('True Labels')
                    st.pyplot(fig, use_container_width=True)

                results_list.append({
                    'Algorithm': name, 'Training Time': training_time, 'Prediction Speed': prediction_time,
                    'Memory Usage': memory_kb, 'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)': eval_metrics_str
                })
    
    # --- K-Means Evaluation (New Style) ---
    if "K-means" in selected_models:
        with st.expander("**K-means Clustering** Results", expanded=True):
            X_processed = preprocessor.fit_transform(X_full)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            
            start_time = time.time()
            cluster_labels = kmeans.fit_predict(X_processed)
            training_time = time.time() - start_time
            silhouette = silhouette_score(X_processed, cluster_labels)

            k_col1, k_col2 = st.columns(2)
            with k_col1:
                st.metric("Silhouette Score", f"{silhouette:.4f}")
                st.info("A Silhouette Score closer to 1 indicates better-defined clusters.")
            
            with k_col2:
                st.markdown("**Computational Cost**")
                st.markdown(f"**⏱️ Training Time:** `{training_time:.4f} seconds`")

            results_list.append({
                'Algorithm': 'K-means', 'Training Time': training_time, 'Prediction Speed': np.nan, 'Memory Usage': np.nan,
                'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)': f"Silhouette Score: {silhouette:.2f}"
            })
            
    # --- Final Summary Table ---
    if results_list:
        st.divider()
        st.header("📜 Final Comparison Summary")
        st.markdown("This table merges the practical performance metrics with theoretical properties of each algorithm for a holistic comparison.")
        
        theoretical_data = {
            'Algorithm': ["Decision Tree", "Logistic Regression", "k-Nearest Neighbors (kNN)", "Support Vector Machine (SVM)", "Neural Networks (NNs)", "Random Forest", "K-means"],
            'Bias–Variance': ["Low Bias, High Variance", "High Bias, Low Variance", "Low Bias, High Variance", "Tunable", "Low Bias, Medium Variance", "Low Bias, High Variance", "N/A"],
            'Data Size Sensitivity': ["Moderate", "Low", "High", "High", "Low", "Low", "Low"],
            'Hyperparameter Sensitivity': ["High", "Low", "High", "Very High", "Very High", "Moderate", "High"],
            'Robustness': ["Moderate", "Low", "Low", "High", "Moderate", "Very High", "Low"],
            'Best-Suited Metrics': ["Accuracy, F1-Score", "Accuracy, AUC-ROC", "Accuracy, F1-Score", "Accuracy, F1-Score", "Accuracy, Log-Loss", "Accuracy, F1-Score", "Silhouette Score"],
            'Remarks': ["Easy to interpret", "Simple & fast", "Slow for prediction", "Powerful but slow to train", "Flexible 'black box'", "Robust & accurate", "Unsupervised grouping"]
        }
        
        results_df = pd.DataFrame(results_list)
        theoretical_df = pd.DataFrame(theoretical_data)
        
        final_df = pd.merge(theoretical_df, results_df, on='Algorithm', how='inner')
        
        # Reorder columns to exactly match the desired format
        final_df = final_df[[
            'Algorithm', 'Bias–Variance', 'Data Size Sensitivity', 'Training Time', 'Prediction Speed',
            'Memory Usage', 'Hyperparameter Sensitivity', 'Robustness',
            'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)', 'Best-Suited Metrics', 'Remarks'
        ]]
        
        # Filter the final_df to only show selected models
        final_df = final_df[final_df['Algorithm'].isin(selected_models)]

        st.dataframe(final_df, use_container_width=True)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=convert_df_to_csv(final_df),
            file_name="ml_algorithm_comparison.csv",
            mime="text/csv",
        )

else:
    st.info("👈 Please select one or more models from the sidebar to begin the analysis.")
