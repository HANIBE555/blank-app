import streamlit as st
import numpy as np
import joblib

# טווחי מינימום ומקסימום לערכים לנרמול
min_max_values = {
    "tumor-size": (2, 52),
    "inv-nodes": (1, 25),
    "deg-malig": (1, 3)
}

def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

st.title("🔬 תחזית חזרת סרטן - הזנת נתונים ונרמול")

# טעינת המודל והעמודות
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# יצירת קלט מהמשתמש
tumor_size = st.number_input("tumor-size", min_value=2.0, max_value=52.0, value=2.0)
inv_nodes = st.number_input("inv-nodes", min_value=1.0, max_value=25.0, value=1.0)
node_caps = st.selectbox("node-caps", options=[0, 1])
deg_malig = st.number_input("deg-malig", min_value=1, max_value=3, step=1, value=1)
# שאר משתנים בינאריים
breast_quad_central = st.selectbox("breast-quad_central", options=[0,1])
breast_quad_left_low = st.selectbox("breast-quad_left_low", options=[0,1])
breast_quad_left_up = st.selectbox("breast-quad_left_up", options=[0,1])
breast_quad_right_low = st.selectbox("breast-quad_right_low", options=[0,1])
breast_quad_right_up = st.selectbox("breast-quad_right_up", options=[0,1])

# נרמול משתנים רציפים
tumor_size_norm = min_max_normalize(tumor_size, *min_max_values["tumor-size"])
inv_nodes_norm = min_max_normalize(inv_nodes, *min_max_values["inv-nodes"])
deg_malig_norm = min_max_normalize(deg_malig, *min_max_values["deg-malig"])

# יצירת מערך הקלט לפי סדר העמודות במודל
input_data = [
    tumor_size_norm,
    inv_nodes_norm,
    node_caps,
    deg_malig_norm,
    breast_quad_central,
    breast_quad_left_low,
    breast_quad_left_up,
    breast_quad_right_low,
    breast_quad_right_up
]

if st.button("🔍 חשב תחזית"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.error("🔴 סיכון לחזרת סרטן (1)")
    else:
        st.success("🟢 ללא חזרת סרטן (0)")
