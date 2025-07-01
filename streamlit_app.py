import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

st.title("🔬 חיזוי חזרת סרטן עם Logistic Regression + SMOTE + K-Fold")

uploaded_file = st.file_uploader("העלה קובץ CSV עם הנתונים", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Class" not in df.columns:
        st.error("הקובץ חייב להכיל עמודת Class")
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            model = LogisticRegression(max_iter=200, random_state=42)
            model.fit(X_train_res, y_train_res)

            y_pred = model.predict(X_test)

            if i == 5:  # מציג את התוצאות של הקיפול האחרון
                st.write(f"--- Evaluation on Fold {i} ---")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
else:
    st.info("אנא העלה קובץ CSV כדי להתחיל באימון המודל")
