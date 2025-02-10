# Tool-to-analyze-resumes-using-NLP-Techniques-
a Python-based tool that analyzes resumes, extracts key skills, and matches them to job descriptions using NLP techniques.
import os
import re
import spacy
import PyPDF2
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined skill set (Can be expanded)
skill_set = {"Python", "Machine Learning", "NLP", "Data Analysis", "SQL", "Java", "Deep Learning", "TensorFlow", "Keras", "Flask", "Django"}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_skills(text):
    """Extract skills from resume text using NLP."""
    doc = nlp(text)
    skills = {token.text for token in doc.ents if token.label_ == "ORG" or token.text in skill_set}
    return list(skills)

def calculate_similarity(resume_text, job_desc):
    """Compute similarity score between resume and job description."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100, 2)

def main():
    """Main program loop."""
    resume_path = input("Enter path to resume (PDF/DOCX): ").strip()
    job_desc = input("Paste the job description: ").strip()
    
    if not os.path.exists(resume_path):
        print("Error: File not found!")
        return
    
    # Extract text from resume
    if resume_path.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_path)
    elif resume_path.endswith(".docx"):
        resume_text = extract_text_from_docx(resume_path)
    else:
        print("Error: Unsupported file format!")
        return
    
    # Extract skills from resume
    skills = extract_skills(resume_text)
    print("\nExtracted Skills:", skills)
    
    # Calculate similarity score
    similarity_score = calculate_similarity(resume_text, job_desc)
    print(f"\nResume Match Score: {similarity_score}%")
    
    # Save results (optional)
    result_df = pd.DataFrame({"Skills": [', '.join(skills)], "Match Score": [similarity_score]})
    result_df.to_csv("resume_match_results.csv", index=False)
    print("\nResults saved to 'resume_match_results.csv'")

if __name__ == "__main__":
    main()

