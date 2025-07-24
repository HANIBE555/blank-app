import streamlit as st
from PIL import Image

# יישור לימין
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# לוגו של סרטן השד (מומלץ לשים קובץ בשם breast_cancer_logo.png בתיקיית הפרויקט)
image = Image.open("breast_cancer_logo.png")
st.image(image, width=150)

# כותרת
st.title("ברוכים הבאים למערכת ניבוי חזרת סרטן השד")

# תיאור קצר
st.markdown("""
מערכת זו נועדה לסייע לרופאים בקבלת החלטות קליניות, 
באמצעות שני מודולים עיקריים:
""")

# עיצוב 2 כפתורים במרכז
col1, col2 = st.columns(2)

with col1:
    if st.button("🔬 חיזוי סיכוי לחזרת מחלה"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")

with col2:
    if st.button("🖼️ בדיקת CT"):
        st.switch_page("pages/2_בדיקת_CT.py")
