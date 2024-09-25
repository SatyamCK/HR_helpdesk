import os
import pytesseract
from PIL import Image
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from prompt import *

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the Gemini model
# model = ChatGoogleGenerativeAI(
#     model="gemini-pro",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2,
#     convert_system_message_to_human=True
# )
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
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
def generate_summary(text,prompt):
    try:
        if not text.strip():
            return "No text found in the image."
        
        
        response = model.predict(prompt)
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
        transaction_summary = ''
        identification_summary = ''
        
        for uploaded_file in uploaded_files:
            # Open and process each uploaded file
            image = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image)

            # Prepare prompts
            identification_prompt = prompt1 + f"Here is the extracted text:\n\n{extracted_text}\n\n"
            transaction_prompt = prompt2 + f"Here is the extracted text:\n\n{extracted_text}\n\n"
            check_doc_type_prompt = prompt3 + f"Here is the extracted text:\n\n{extracted_text}\n\n"

            # Determine the document type
            doc_type = generate_summary(extracted_text, check_doc_type_prompt).strip()

            if doc_type == 'Payslip':
                # Generate transaction summary for Payslip
                transaction_summary = generate_summary(extracted_text, transaction_prompt).strip()
            elif doc_type == '':
                # Generate identification summary for Driver License
                identification_summary += generate_summary(extracted_text, identification_prompt).strip() + '\n'  # Append for multiple images

        # Print summaries if they are not empty
        if identification_summary:
            st.subheader("Identification Summary")
            st.write(identification_summary)
        else:
            st.write("No identification summary generated.")
        st.write("")
        if transaction_summary:
            st.subheader("Transaction Summary")
            st.write(transaction_summary)
        else:
            st.write("No transaction summary generated.")
