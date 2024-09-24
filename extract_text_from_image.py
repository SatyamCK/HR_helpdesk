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
    temperature=0.2,
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
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        # print(text)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Function to generate summary using the model
def generate_summary(text):
    try:
        if not text.strip():
            return "No text found in the image."
        
        prompt = f"Please generate a detailed and accurate summary. Focus on capturing the essential information, key insights, and any notable themes present in the text. Ensure that the summary is clear and easy to understand, providing context where necessary. Here is the extracted text:\n\n{text}"
        prompt2 = f"""
            1. Generate a detailed summary based on the extracted text from the image.
            2. Focus Areas:
                i) Capture essential information and key insights.
                ii) Highlight any notable themes present in the text.
            3. Ensure that the summary is clear and easy to understand.
            4. Provide additional context where necessary to enhance comprehension.
            Here is the extracted text:\n\n{text}
        """
        response = model.predict(prompt2)
        return response
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "An error occurred while generating the summary."

st.markdown("""
<style>
    .stSidebar {
        background-color: #1900ff3d !important;
    }
    """, unsafe_allow_html=True)

st.sidebar.image('./images/logo.png', width=300)

# Upload multiple images
uploaded_files = st.file_uploader("Upload image files (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button('Generate Summary'):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Open and process each uploaded file
            image = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image)

            # Display the title for each image
            st.subheader(f"Summary for {uploaded_file.name}")

            # Generate and display the summary for the current image
            summary = generate_summary(extracted_text)
            st.write(summary)
