import pypdf
import os

pdf_path = r'e:\AI_Chilla\AI_CHILLA_2026\Kaggle_datasets\Rice\Prod_Plan_Rice_2025.pdf'
output_path = r'e:\AI_Chilla\AI_CHILLA_2026\Kaggle_datasets\Rice\extracted_text.txt'

def extract_text(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_text(pdf_path, output_path)
