import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic + SMOTE + K-Fold")

# שלב 1: אם אין מודל – העלאת קובץ ואימון
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.subheader("📁 העלאת קובץ נתונים")
    uploaded_file = st.file_uploader("העלה את הקובץ final_data_for_project.csv", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Class" not in df.columns:
            st.error("הקובץ חייב להכיל עמודת Class.")
        else:
            X = df.drop("Class", axis=1)
            y = df["Class"]

            # קידוד כל העמודות הקטגוריות
            X = pd.get_dummies(X)

            # בדיקות ניקיון נתונים
            if X.isnull().sum().sum() > 0:
                st.error("הקובץ מכיל ערכים חסרים. נא לנקות לפני האימון.")
            else:
                X = X.astype('float32')  # המרה כדי למנוע שגיאות טיפוס

                model = Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('logistic', LogisticRegression(max_iter=200))
                ])
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_tra_
