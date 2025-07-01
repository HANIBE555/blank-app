import streamlit as st
import joblib

# טווחים לנרמול
min_max_values = {
    "tumor-size": (2, 52),
    "inv-nodes": (1, 25),
    "deg-malig": (1, 3)
}
# הצגת הפיצ'רים שהמודל דורש (לבדיקה בלבד)
features = joblib.load("features.pkl")
st.write("🔍 הפיצ'רים שהמודל דורש:", features)
st.write("📊 סה\"כ פיצ'רים שמצופים מהמודל:", len(features))
st.write("📥 סה\"כ פיצ'רים שנמסרו בפועל:", len(input_data))


# פונקציית נרמול עם אזהרות
def min_max_normalize(value, min_val, max_val, field_name=""):
    if value < min_val:
        st.warning(f"⚠️ הערך {value} ב־'{field_name}' קטן מהמינימום האפשרי ({min_val}) – מנורמל ל־0")
        return 0.0
    elif value > max_val:
        st.warning(f"⚠️ הערך {value} ב־'{field_name}' גדול מהמקסימום האפשרי ({max_val}) – מנורמל ל־1")
        return 1.0
    return (value - min_val) / (max_val - min_val)

# פונקציה להפיכת "כן"/"לא" ל־0/1
def to_binary(val):
    return 1 if val == "כן" else 0

# כותרת והסבר כללי
st.title("🔬 תחזית חזרת סרטן - הזנת נתונים לרופא")
st.markdown("""
🧑‍⚕️ **הנחיות להזנת ערכים:**
- `tumor-size` ו־`inv-nodes`: נא להזין את **אמצע הטווח** (למשל טווח 10–14 → הזן 12).
- משתנים בינאריים (כן/לא):  
  הזן **"כן" = 1**, **"לא" = 0** דרך תיבת הבחירה.""")

# טעינת המודל ורשימת הפיצ'רים
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# קלטים רציפים עם הסברים
tumor_size = st.number_input("tumor-size (אמצע טווח בגודל הגידול)", step=0.1)
inv_nodes = st.number_input("inv-nodes (אמצע טווח במספר קשריות נגועות)", step=0.1)
deg_malig = st.number_input("deg-malig (דרגת ממאירות – ערך שלם בלבד)", step=1)


# קלטים בינאריים (כן/לא) עם הסבר
node_caps = st.selectbox("node-caps (קופסית קשרית נגועה: כן = 1, לא = 0)", options=["לא", "כן"])
breast_quad_central = st.selectbox("breast-quad_central (גידול במרכז השד: כן = 1, לא = 0)", options=["לא", "כן"])
breast_quad_left_low = st.selectbox("breast-quad_left_low (גידול בשד שמאל תחתון: כן = 1, לא = 0)", options=["לא", "כן"])
breast_quad_left_up = st.selectbox("breast-quad_left_up (גידול בשד שמאל עליון: כן = 1, לא = 0)", options=["לא", "כן"])
breast_quad_right_low = st.selectbox("breast-quad_right_low (גידול בשד ימין תחתון: כן = 1, לא = 0)", options=["לא", "כן"])
breast_quad_right_up = st.selectbox("breast-quad_right_up (גידול בשד ימין עליון: כן = 1, לא = 0)", options=["לא", "כן"])

# גיל המעבר - קידוד one-hot עם הסבר
menopause_choice = st.radio("מצב גיל המעבר:", ["ge40 (מעל גיל 40)", "lt40 (מתחת לגיל 40)", "premeno (לפני גיל מעבר)"])
menopause_ge40 = 1 if menopause_choice.startswith("ge40") else 0
menopause_lt40 = 1 if menopause_choice.startswith("lt40") else 0
menopause_premeno = 1 if menopause_choice.startswith("premeno") else 0

# נרמול משתנים רציפים
tumor_size_norm = min_max_normalize(tumor_size, *min_max_values["tumor-size"], field_name="tumor-size")
inv_nodes_norm = min_max_normalize(inv_nodes, *min_max_values["inv-nodes"], field_name="inv-nodes")
deg_malig_norm = min_max_normalize(deg_malig, *min_max_values["deg-malig"], field_name="deg-malig")

# יצירת מערך קלט למודל לפי סדר העמודות
input_data = [
    tumor_size_norm,
    inv_nodes_norm,
    to_binary(node_caps),
    deg_malig_norm,
    to_binary(breast_quad_central),
    to_binary(breast_quad_left_low),
    to_binary(breast_quad_left_up),
    to_binary(breast_quad_right_low),
    to_binary(breast_quad_right_up),
    menopause_ge40,
    menopause_lt40,
    menopause_premeno
]

# תחזית
if st.button("🔍 חשב תחזית"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.error("🔴 סיכון לחזרת סרטן (1)")
    else:
        st.success("🟢 ללא חזרת סרטן (0)")
