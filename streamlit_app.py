import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic + SMOTE + K-Fold")

def prepare_data(df):
    df = df.dropna()
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X = pd.get_dummies(X)
    X = X.replace([float('inf'), -float('inf')], 0)  # להחליף אינסוף באפס

    # המרה למספרים עם טיפול בשגיאות
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    if X.isnull().sum().sum() > 0:
        st.error("יש ערכים חסרים אחרי המרה מספרית. אנא נקו את הנתונים.")
        st.stop()

    X = X.astype(float)
    y = y.astype(int)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.subheader("📁 העלאת קובץ נתונים")
    uploaded_file = st.file_uploader("העלה קובץ CSV עם עמודת Class", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Class" not in df.columns:
            st.error("❌ הקובץ חייב להכיל עמודת Class.")
        else:
            X, y = prepare_data(df)

            smote = SMOTE(random_state=42)
            model = LogisticRegression(max_iter=200)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # הדפסות מידע לאבחון
                st.write(f"--- קיפול {i} ---")
                st.write("Shape X_train:", X_train.shape)
                st.write("Types X_train:", X_train.dtypes)
                st.write("Any NaNs in X_train:", X_train.isnull().any().any())
                st.write("Shape y_train:", y_train.shape)
                st.write("Types y_train:", y_train.dtype)
                st.write("Any NaNs in y_train:", y_train.isnull().any())

                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)

                if i == 5:
                    st.write(f"‏‏דוח קיפול מס' {i}")
                    st.text(confusion_matrix(y_test, y_pred))
                    st.text(classification_report(y_test, y_pred))

            joblib.dump(model, MODEL_PATH)
            joblib.dump(X.columns.tolist(), FEATURES_PATH)

            st.success("✅ המודל אומן ונשמר. רענני את הדף לצורך תחזית.")

else:
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
