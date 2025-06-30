import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.title("🔬 חיזוי חזרת סרטן עם Logistic + SMOTE")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.subheader("📁 העלאת קובץ נתונים")
    uploaded_file = st.file_uploader("העלה קובץ CSV עם עמודת Class", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Class" not in df.columns:
            st.error("❌ הקובץ חייב להכיל עמודת Class.")
        else:
            X = df.drop("Class", axis=1)
            y = df["Class"]

            st.write("טיפוסי העמודות ב-X לפני get_dummies:")
            st.write(X.dtypes)

            X = pd.get_dummies(X)
            X = X.fillna(0)
            X = X.astype(float)
            y = y.astype(int)

            st.write("טיפוסי העמודות ב-X אחרי get_dummies:")
            st.write(X.dtypes)
            st.write("טיפוס y:", y.dtype)

            if X.isnull().sum().sum() > 0:
                st.error("❌ יש ערכים חסרים בנתונים!")
                st.stop()

            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            st.write(f"Shape of X: {X.shape}")
            st.write(f"Shape of y: {y.shape}")

            smote = SMOTE(random_state=42)
            try:
                X_res, y_res = smote.fit_resample(X, y)
                st.success(f"נתונים מאזנים, X_res shape: {X_res.shape}, y_res shape: {y_res.shape}")
            except Exception as e:
                st.error(f"❌ שגיאה ב-SMOTE: {e}")
                st.stop()

            model = LogisticRegression(max_iter=200)
            model.fit(X_res, y_res)

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
