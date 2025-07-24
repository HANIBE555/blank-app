import streamlit as st
from PIL import Image
import base64

# הגדרת רקע
def set_bg(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
    encoded = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# יישום הרקע
set_bg("IMG.png")  # או את הנתיב המלא אם מריץ מקומית

# הגדרות עיצוב כלליות לימין
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
</style>
