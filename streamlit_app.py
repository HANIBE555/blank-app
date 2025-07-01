import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic Regression + K-Fold")

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
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            if i == 5:
                st.write(f"--- תוצאות קיפול מספר {i} ---")
                st.write("מטריצת בלבול:")
                st.write(cm)
                st.write("דו\"ח סיווג:")
                st.text(report)

        # שמירת המודל והעמודות לשימוש עתידי
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
        st.success("✅ המודל אומן ונשמר בהצלחה.")
else:
    st.info("אנא העלה קובץ CSV עם עמודת Class כדי להתחיל את האימון.")
