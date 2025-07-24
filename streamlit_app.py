import streamlit as st
st.markdown(
    """
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .css-1v0mbdj.e115fcil1 {  /* כותרת */
        direction: rtl;
        text-align: right;
    }
    .stMarkdown p {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox div, .stRadio div, .stNumberInput label {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
from sklearn.linear_model import LogisticRegression

# טווחים לנרמול
min_max_values = {
    "tumor-size": (2, 52),
    "inv-nodes": (1, 25),
    "deg-malig": (1, 3)
}

# פונקציית נרמול שקטה
def min_max_normalize(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

# המרה של "כן"/"לא" ל־0/1
def to_binary(val):
    return 1 if val == "כן" else 0

# כותרת והנחיות
st.title("🔬 תחזית חזרת סרטן - הזנת נתונים לרופא")
st.markdown("""
🧑‍⚕ *הנחיות להזנת ערכים:*
- tumor-size ו־inv-nodes: יש להזין את *אמצע הטווח* (למשל טווח 10–14 → הזן 12).
- משתנים בינאריים (כן/לא): הזן *"כן" = 1, **"לא" = 0* דרך תיבת הבחירה.
- deg-malig: ערך מספרי שלם בין 1 ל־3.
""")

# העלאת קובץ CSV ואימון מודל
uploaded_file = st.file_uploader("📁 העלה קובץ CSV עם עמודת Class", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Class" not in df.columns:
        st.error("❌ הקובץ חייב לכלול עמודה בשם 'Class'")
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        st.session_state.model = model
        st.success("🤖 המודל אומן ונשמר בזיכרון")

# בדיקה אם יש מודל
if "model" in st.session_state:
    model = st.session_state.model

    # קלטים רציפים
    tumor_size = st.number_input("tumor-size (אמצע טווח בגודל הגידול)", step=0.1)
    inv_nodes = st.number_input("inv-nodes (אמצע טווח במספר קשריות נגועות)", step=0.1)
    deg_malig = st.number_input("deg-malig (דרגת ממאירות – ערך שלם בלבד)", step=1)

    # משתנים בינאריים
    node_caps = st.selectbox("node-caps (קופסית קשרית נגועה)", options=["לא", "כן"])
    irradiat = st.selectbox("irradiat (טופל בהקרנות)", options=["לא", "כן"])

    # גיל מעבר
    menopause_choice = st.radio("מצב גיל המעבר:", ["ge40 (מעל גיל 40)", "lt40 (מתחת לגיל 40)", "premeno (לפני גיל מעבר)"])
    menopause_ge40 = 1 if menopause_choice.startswith("ge40") else 0
    menopause_lt40 = 1 if menopause_choice.startswith("lt40") else 0
    menopause_premeno = 1 if menopause_choice.startswith("premeno") else 0

    # מיקום הגידול
    breast_quad_central = st.selectbox("breast-quad_central (גידול במרכז השד)", options=["לא", "כן"])
    breast_quad_left_low = st.selectbox("breast-quad_left_low (גידול בשד שמאל תחתון)", options=["לא", "כן"])
    breast_quad_left_up = st.selectbox("breast-quad_left_up (גידול בשד שמאל עליון)", options=["לא", "כן"])
    breast_quad_right_low = st.selectbox("breast-quad_right_low (גידול בשד ימין תחתון)", options=["לא", "כן"])
    breast_quad_right_up = st.selectbox("breast-quad_right_up (גידול בשד ימין עליון)", options=["לא", "כן"])

    # נרמול משתנים רציפים
    tumor_size_norm = min_max_normalize(tumor_size, *min_max_values["tumor-size"])
    inv_nodes_norm = min_max_normalize(inv_nodes, *min_max_values["inv-nodes"])
    deg_malig_norm = min_max_normalize(deg_malig, *min_max_values["deg-malig"])

    # יצירת מערך קלט לפי סדר המודל (13 פיצ'רים)
    input_data = [
        tumor_size_norm,
        inv_nodes_norm,
        to_binary(node_caps),
        deg_malig_norm,
        to_binary(irradiat),
        menopause_ge40,
        menopause_lt40,
        menopause_premeno,
        to_binary(breast_quad_central),
        to_binary(breast_quad_left_low),
        to_binary(breast_quad_left_up),
        to_binary(breast_quad_right_low),
        to_binary(breast_quad_right_up)
    ]

    # תחזית
    if st.button("🔍 חשב תחזית"):
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("🔴 סיכון לחזרת סרטן (1)")
        else:
            st.success("🟢 ללא חזרת סרטן (0)")

else:
    st.info("⬆ יש להעלות קובץ כדי לאמן את המודל לפני הזנת תחזית")
