import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic Regression + SMOTE + K-Fold")

uploaded_file = st.file_uploader("📁 העלה קובץ CSV עם עמודת Class בלבד", type="csv")

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
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # כאן הוספתי random_state
        smote = SMOTE(random_state=42)  # וכאן גם

        for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # החלת SMOTE רק על סט האימון
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)

            if i == 5:
                st.write(f"--- תוצאות קיפול מספר {i} ---")
                st.write("מטריצת בלבול:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("דו\"ח סיווג:")
                st.text(classification_report(y_test, y_pred))

        # שמירת המודל והעמודות לשימוש עתידי
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
        st.success("✅ המודל אומן ונשמר בהצלחה.")
else:
    st.info("אנא העלה קובץ CSV עם עמודת Class כדי להתחיל את האימון.")
