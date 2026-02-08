import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
from io import BytesIO
import base64

# =========================
# File reading functions
# =========================
def read_text(file):
    return file.getvalue().decode("utf-8")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# =========================
# Stopwords filtering
# =========================
def filter_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopwords = set(STOPWORDS).union(set(additional_stopwords))
    filtered = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered)

# =========================
# Download helper
# =========================
def get_image_download_link(buffered, format_="png"):
    base64_img = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{base64_img}" download="wordcloud.{format_}">📥 Download Word Cloud</a>'

# =========================
# Streamlit App
# =========================
st.title("📊 Word Cloud Generator")
st.subheader("Upload a TXT, PDF, or Word document to generate a Word Cloud.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Read file based on MIME type
    if uploaded_file.type == "text/plain":
        text = read_text(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                "application/msword"]:
        text = read_docx(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
        st.stop()

    if not text.strip():
        st.error("❌ The file is empty. Cannot generate Word Cloud.")
    else:
        # Optional: add extra stopwords here
        additional_stopwords = []
        text_filtered = filter_stopwords(text, additional_stopwords)

        # Generate Word Cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            collocations=False
        ).generate(text_filtered)

        # Display Word Cloud
        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Create download link for Word Cloud image
        buffered = BytesIO()
        fig.savefig(buffered, format="png", bbox_inches='tight')
        st.markdown(get_image_download_link(buffered), unsafe_allow_html=True)