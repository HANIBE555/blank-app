import streamlit as st
import joblib

# טעינת המודל מקובץ שמגיע עם הפרויקט (לא מאמן מחדש)
model = joblib.load("model.pkl")
