import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

st.title("🔬 Logistic Regression עם K-Fold עקבי (shuffle + random_state)")

uploaded_file = st.file_uploader("העלה קובץ CSV עם עמודת 'Class'", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("הקובץ חייב להכיל עמודת 'Class'.")
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # כאן חשוב להגדיר shuffle=True ו-random_state קבוע
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=200, random_state=42)

        for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"--- קיפול {i} ---")
            st.write("מטריצת בלבול:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("דו\"ח סיווג:")
            st.text(classification_report(y_test, y_pred))
