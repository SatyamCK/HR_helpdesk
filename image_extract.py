import os
import pytesseract
from PIL import Image
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5,
    convert_system_message_to_human=True
)

# Streamlit configuration
st.set_page_config(
    page_title="Image Summarizer Bot",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to extract text from an image
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Function to generate summary using the model
def generate_summary(text):
    try:
        if not text.strip():
            return "No text found in the image."
        
        prompt = f"Summarize the following text in detailed manner:\n\n{text}"

        response = model.predict(prompt)
        return response
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "An error occurred while generating the summary."

st.markdown("""
<style>
    .stSidebar {
        background-color: #9896AA !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image('./images/logo.png', width=300)

# Upload image
uploaded_file = st.file_uploader("Upload an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Save uploaded file to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the image
    extracted_text = extract_text_from_image("temp_image.png")
    
    # Generate and display the summary
    if st.button("Generate Summary"):
        summary = generate_summary(extracted_text)
        st.subheader("Generated Summary")
        st.write(summary)
