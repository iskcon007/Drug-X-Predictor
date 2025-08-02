# streamlit: Used to build the interactive web app.
# pandas: For data manipulation.
# matplotlib.pyplot & seaborn: For plotting graphs.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# These are all scikit-learn tools:
# train_test_split: Splits dataset into training/testing.
# # LabelEncoder: Converts categories (like 'Male', 'Female') to numbers.
#The rest are ML models and metrics.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Suppresses any warnings to keep the app clean.
import warnings
warnings.filterwarnings("ignore")

#Sets the app title and instructions for the user.
st.set_page_config(page_title="Custom Drug Classifier", layout="wide")
st.title("🧠 Multi-Drug Use Classifier")
st.markdown("Upload your drug dataset and select any drug (e.g., Alcohol, Meth, Cannabis) to analyze usage patterns.")

#Lets user upload a .csv file.
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

#Loads the file and shows the first 10 rows for user reference.
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Detect drug columns by CL0–CL6 which represent the drug level 
    def is_drug_column(col):
        unique_vals = df[col].dropna().unique()
        return set(unique_vals).issubset({'CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'})
    #Filters and returns the drug-related columns.
    drug_cols = [col for col in df.columns if is_drug_column(col)]

    #Shows an error if no drug columns are found.
    if not drug_cols:
        st.error("❌ No valid drug columns found (CL0–CL6 format expected).")
    else:
        selected_drug = st.sidebar.selectbox("🔬 Select a Drug to Analyze", drug_cols)

        # Converts 7-class drug levels into a binary class:
            # CL0–CL2 → Non-User (0)
            # CL3–CL6 → User (1)
        df[f"{selected_drug}_Binary"] = df[selected_drug].apply(lambda x: 0 if x in ['CL0', 'CL1', 'CL2'] else 1)

        # Drop selected drug (and raw drug column) from features
        working_df = df.drop(columns=[selected_drug])

        # Encode categoricals
        for col in working_df.columns:
            if working_df[col].dtype == 'object':
                le = LabelEncoder()
                working_df[col] = le.fit_transform(working_df[col].astype(str))

        # Feature/target split
        X = working_df.drop(columns=[f"{selected_drug}_Binary"])
        y = working_df[f"{selected_drug}_Binary"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define tuned models
        models = {
            "Logistic Regression": LogisticRegression(C=0.5, solver='liblinear'),
            "SVM (RBF Kernel)": SVC(C=1.0, kernel='rbf', gamma='scale'),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion='entropy'),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        }

        # Model training
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "y_pred": y_pred,
                "report": classification_report(y_test, y_pred, output_dict=True),
                "confusion": confusion_matrix(y_test, y_pred)
            }

        # Accuracy bar chart
        st.subheader(f"📊 Accuracy Comparison for: {selected_drug}")
        acc_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy (%)": [round(v["accuracy"] * 100, 2) for v in results.values()]
        })
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=acc_df, x="Model", y="Accuracy (%)", palette="viridis", ax=ax)
        ax.set_ylim(0, 100)
        for index, row in acc_df.iterrows():
            ax.text(index, row["Accuracy (%)"] + 1, f"{row['Accuracy (%)']}%", ha='center', fontweight='bold')
        ax.set_title(f"Model Accuracy on Predicting {selected_drug} Use")
        st.pyplot(fig)

        # Choose a model to inspect
        selected_model = st.selectbox("🔍 Select a model to view report", list(results.keys()))

        # Classification Report
        st.subheader(f"📋 Classification Report - {selected_model}")
        report_df = pd.DataFrame(results[selected_model]["report"]).transpose()
        st.dataframe(report_df.style.format({
            'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'
        }).background_gradient(cmap='Blues'), use_container_width=True)

        # Confusion Matrix
        st.subheader(f"🧩 Confusion Matrix - {selected_model}")
        cm = results[selected_model]["confusion"]
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_xticklabels(['Non-User', 'User'])
        ax_cm.set_yticklabels(['Non-User', 'User'], rotation=0)
        st.pyplot(fig_cm)

        # Final Verdict
        user_count = sum(results[selected_model]['y_pred'])
        non_user_count = len(results[selected_model]['y_pred']) - user_count
        st.subheader("📈 Final Verdict")
        if user_count > non_user_count:
            st.success(f"More predicted as **USERS** ({user_count}) than non-users ({non_user_count}).")
        elif non_user_count > user_count:
            st.success(f"More predicted as **NON-USERS** ({non_user_count}) than users ({user_count}).")
        else:
            st.info("Equal number of users and non-users predicted.")
