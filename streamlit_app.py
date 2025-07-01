import streamlit as st
import numpy as np
import joblib

# טווחי מינימום ומקסימום לנרמול
min_max_values = {
    "tumor-size": (2, 52),
    "inv-nodes": (1, 25),
    "deg-malig": (1, 3)
}

# פונקציית נרמול עם טיפול בחריגים
def min_max_normalize(value, min_val, max_val):
    if value < min_val:
        st.warning(f"⚠️ הערך {value} קטן מהמינימום האפשרי ({min_val}) - מנורמל ל-0")
        return 0.0
    elif value > max_val:
        st.warning(f"⚠️ הערך {value} גדול מהמקסימום האפשרי ({max_val}) - מנורמל ל-1")
        return 1.0
    return (value - min_val) / (max_val - min_val)

st.title("🔬 תחזית חזרת סרטן - הזנת נתונים ונרמול")

# טעינת המודל ורשימת העמודות
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# הזנת נתונים חופשית
tumor_size = st.number_input("tumor-size", step=0.01)
inv_nodes = st.number_input("inv-nodes", step=0.01)
deg_malig = st.number_input("deg-malig", step=1)

# משתנים בינאריים
node_caps = st.selectbox("node-caps", options=[0, 1])
breast_quad_central = st.selectbox("breast-quad_central", options=[0,1])
breast_quad_left_low = st.selectbox("breast-quad_left_low", options=[0,1])
breast_quad_left_up = st.selectbox("breast-quad_left_up", options=[0,1])
breast_quad_right_low = st.selectbox("breast-quad_right_low", options=[0,1])
breast_quad_right_up = st.selectbox("breast-quad_right_up", options=[0,1])

# נרמול משתנים רציפים
tumor_size_norm = min_max_normalize(tumor_size, *min_max_values["tumor-size"])
inv_nodes_norm = min_max_normalize(inv_nodes, *min_max_values["inv-nodes"])
deg_malig_norm = min_max_normalize(deg_malig, *min_max_values["deg-malig"])

# יצירת מערך קלט למודל
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

# חישוב התחזית
if st.button("🔍 חשב תחזית"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.error("🔴 סיכון לחזרת סרטן (1)")
    else:
        st.success("🟢 ללא חזרת סרטן (0)")
