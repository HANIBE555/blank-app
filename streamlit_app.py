import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic Regression + K-Fold ללא SMOTE")

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

            # המרת קטגוריות ל-Dummies אם יש
            X = pd.get_dummies(X)

            model = LogisticRegression(max_iter=200)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                st.write(f"--- קיפול {i} ---")
                st.write("מטריצת בלבול:")
                st.write(cm)
                st.write("דו"ח סיווג:")
                st.text(report)

            # שמירת המודל והעמודות לשימוש עתידי
            joblib.dump(model, MODEL_PATH)
            joblib.dump(X.columns.tolist(), FEATURES_PATH)
            st.success("✅ המודל אומן ונשמר. רענן את הדף לצורך תחזית.")

else:
    # שלב 2: טעינת מודל ותחזית
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    st.subheader("📝 הזנת תצפית חדשה")
    user_input = []
    for feature in features:
        val = st.number_input(f"{feature}", value=0.0)
        user_input.append(val)

    if st.button("🔍 חשב תחזית"):
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error("🔴 התחזית: סיכון לחזרת סרטן (1)")
        else:
            st.success("🟢 התחזית: ללא חזרת סרטן (0)")
