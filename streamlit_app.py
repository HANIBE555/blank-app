import streamlit as st
import base64

# הגדרת רקע מתמונה
def set_bg(img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: right top;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("IMG.png")

# RTL
st.markdown("""
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: Arial;
    }
    </style>
""", unsafe_allow_html=True)

# תוכן דף הבית
st.title("🎗️ ברוך הבא למערכת הרופא")
st.subheader("אנא בחר את סוג התחזית שתרצה לבצע:")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔬 חיזוי חזרת מחלה"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")

with col2:
    if st.button("🖼️ בדיקת CT"):
        st.switch_page("pages/2_בדיקת_CT.py")
