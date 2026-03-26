from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# -------- CLEAN TEXT FUNCTION (LESS AGGRESSIVE) --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep words, remove symbols
    return text

# -------- READ TXT --------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# -------- READ PDF --------
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# -------- INPUT JOB DESCRIPTION --------
job_description = input("Enter Job Description:\n")

# -------- INPUT FILE NAMES --------
n = int(input("\nEnter number of resume files: "))
files = []

for i in range(n):
    file = input(f"Enter file name {i+1} (with .pdf/.txt): ")
    files.append(file)

resumes = []

# -------- READ FILES + DEBUG --------
for file in files:
    try:
        if file.endswith(".txt"):
            text = read_txt(file)
        elif file.endswith(".pdf"):
            text = read_pdf(file)
        else:
            print(f"Unsupported file format: {file}")
            text = ""

        if not text.strip():
            print(f"⚠️ Warning: {file} has no readable text")

        print(f"\nDEBUG ({file}):\n{text[:200]}\n")  # show first 200 chars
        resumes.append(text)

    except Exception as e:
        print(f"Error reading file: {file}")
        resumes.append("")

# -------- CLEAN DATA --------
job_description = clean_text(job_description)
resumes = [clean_text(r) for r in resumes]

# -------- NLP PROCESSING --------
documents = [job_description] + resumes

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# -------- KEYWORD BOOSTING (IMPORTANT) --------
important_keywords = ["python", "machine learning", "nlp", "data"]

boosted_scores = []
for i, score in enumerate(similarity_scores[0]):
    resume_text = resumes[i]
    boost = 0
    for word in important_keywords:
        if word in resume_text:
            boost += 0.05
    boosted_scores.append(score + boost + 0.01)  # +0.01 avoids zero

# -------- RANKING --------
ranked_resumes = sorted(
    zip(files, boosted_scores),
    key=lambda x: x[1],
    reverse=True
)

# -------- OUTPUT --------
print("\nTop Matching Resumes:\n")
for i, (file, score) in enumerate(ranked_resumes, 1):
    print(f"{i}. {file} → Score: {score:.2f}")