# ============================================
# TASK 02 - RESUME PARSER USING NLP
# TEXT EXTRACTION MODULE
# ============================================

import os
import pdfplumber
from docx import Document


def extract_from_pdf(file_path):
    """
    Extract text from a PDF resume.
    """

    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"PDF extraction error: {e}")

    return text


def extract_from_docx(file_path):
    """
    Extract text from a DOCX resume.
    """

    text = ""

    try:
        document = Document(file_path)

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"

    except Exception as e:
        print(f"DOCX extraction error: {e}")

    return text


def extract_text(file_path):
    """
    Detect file type and extract text.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_from_pdf(file_path)

    elif extension == ".docx":
        return extract_from_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Please upload PDF or DOCX."
        )


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("RESUME TEXT EXTRACTION TEST")
    print("=" * 50)

    file_path = input("Enter resume file path: ").strip()

    if os.path.exists(file_path):

        resume_text = extract_text(file_path)

        print("\nExtracted Resume Text:")
        print("-" * 50)
        print(resume_text)

        print("\n" + "=" * 50)
        print(f"Characters extracted: {len(resume_text)}")
        print("=" * 50)

    else:

        print("\n❌ File not found.")