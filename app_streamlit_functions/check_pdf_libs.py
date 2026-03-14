import sys
try:
    import PyPDF2
    print("PyPDF2 is installed")
except ImportError:
    print("PyPDF2 is not installed")

try:
    import pypdf
    print("pypdf is installed")
except ImportError:
    print("pypdf is not installed")

try:
    import pdfplumber
    print("pdfplumber is installed")
except ImportError:
    print("pdfplumber is not installed")
