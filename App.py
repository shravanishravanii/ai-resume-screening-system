import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# -------- CLEAN TEXT --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

# -------- READ TXT --------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# -------- READ PDF --------
def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
    except:
        pass
    return text

# -------- GLOBAL FILE LIST --------
selected_files = []

# -------- SELECT FILES --------
def select_files():
    global selected_files
    files = filedialog.askopenfilenames(filetypes=[("Resume Files", "*.pdf *.txt")])
    selected_files = files
    file_label.config(text=f"{len(files)} files selected")

# -------- ANALYZE FUNCTION --------
def analyze():
    job_description = job_entry.get("1.0", tk.END).strip()

    if not job_description:
        messagebox.showerror("Error", "Enter job description")
        return

    if not selected_files:
        messagebox.showerror("Error", "Select resume files")
        return

    resumes = []

    for file in selected_files:
        if file.endswith(".txt"):
            text = read_txt(file)
        elif file.endswith(".pdf"):
            text = read_pdf(file)
        else:
            text = ""

        resumes.append(clean_text(text))

    job_description_clean = clean_text(job_description)

    documents = [job_description_clean] + resumes

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # -------- BOOSTING --------
    keywords = ["python", "machine learning", "nlp", "data"]
    final_scores = []

    for i, score in enumerate(similarity_scores[0]):
        boost = 0
        for word in keywords:
            if word in resumes[i]:
                boost += 0.05
        final_scores.append(score + boost + 0.01)

    ranked = sorted(zip(selected_files, final_scores), key=lambda x: x[1], reverse=True)

    # -------- DISPLAY --------
    result_box.delete("1.0", tk.END)

    for i, (file, score) in enumerate(ranked, 1):
        name = file.split("/")[-1]
        result_box.insert(tk.END, f"{i}. {name} → {score:.2f}\n")

# -------- GUI SETUP --------
root = tk.Tk()
root.title("AI Resume Screening System")
root.geometry("700x600")

tk.Label(root, text="Job Description:").pack(anchor="w", padx=10)

job_entry = tk.Text(root, height=6, width=80)
job_entry.pack(padx=10, pady=5)

tk.Button(root, text="Select Resume Files", command=select_files).pack(pady=10)

file_label = tk.Label(root, text="No files selected")
file_label.pack()

tk.Button(root, text="Analyze Resumes", command=analyze, bg="green", fg="white").pack(pady=10)

tk.Label(root, text="Results:").pack(anchor="w", padx=10)

result_box = tk.Text(root, height=20, width=80)
result_box.pack(padx=10, pady=5)

root.mainloop()