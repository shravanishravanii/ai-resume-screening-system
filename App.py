import streamlit as st
import re
import json
import sqlite3
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -------- OPTIONAL IMPORTS (graceful fallback) --------
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_SUPPORT = True
except ImportError:
    SKLEARN_SUPPORT = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_SUPPORT = True
except Exception:
    SPACY_SUPPORT = False

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="ResumeAI Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- CUSTOM CSS --------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }

    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252842);
        border: 1px solid #3a3f6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }

    .metric-card h2 {
        color: #7c83ff;
        font-size: 2rem;
        margin: 0;
    }

    .metric-card p {
        color: #8b8fa8;
        margin: 4px 0 0 0;
        font-size: 0.85rem;
    }

    .candidate-card {
        background: #1a1d2e;
        border-left: 4px solid #7c83ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    .candidate-card.shortlisted {
        border-left-color: #4caf7d;
    }

    .candidate-card.rejected {
        border-left-color: #e05c5c;
    }

    .tag {
        display: inline-block;
        background: #252842;
        border: 1px solid #3a3f6e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        margin: 3px 3px 3px 0;
        color: #a0a4c0;
    }

    .tag.matched {
        background: #1a3a2a;
        border-color: #4caf7d;
        color: #4caf7d;
    }

    .tag.missing {
        background: #3a1a1a;
        border-color: #e05c5c;
        color: #e05c5c;
    }

    .score-bar-container {
        background: #252842;
        border-radius: 6px;
        height: 8px;
        margin: 6px 0;
    }

    .score-bar {
        background: linear-gradient(90deg, #7c83ff, #4caf7d);
        border-radius: 6px;
        height: 8px;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        color: #7c83ff;
        border-bottom: 1px solid #252842;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c83ff, #5c63df);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 10px 24px;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(124, 131, 255, 0.4);
    }

    .stTextArea textarea, .stTextInput input {
        background: #1a1d2e !important;
        border: 1px solid #3a3f6e !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }

    .sidebar .sidebar-content {
        background: #0d0f1a;
    }

    .history-item {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        cursor: pointer;
        border: 1px solid #252842;
    }

    .duplicate-warning {
        background: #3a2a1a;
        border: 1px solid #e08c3c;
        border-radius: 8px;
        padding: 12px;
        color: #e08c3c;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# -------- DATABASE --------
def init_db():
    conn = sqlite3.connect("screening_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            job_title TEXT,
            num_resumes INTEGER,
            top_candidate TEXT,
            results_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_session(job_title, num_resumes, top_candidate, results):
    conn = sqlite3.connect("screening_history.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions (timestamp, job_title, num_resumes, top_candidate, results_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        job_title,
        num_resumes,
        top_candidate,
        json.dumps(results)
    ))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("screening_history.db")
    c = conn.cursor()
    c.execute("SELECT id, timestamp, job_title, num_resumes, top_candidate FROM sessions ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows


# -------- TEXT EXTRACTION --------
def read_pdf(file):
    if not PDF_SUPPORT:
        return ""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    except Exception as e:
        st.warning(f"PDF read error: {e}")
    return text

def read_docx(file):
    if not DOCX_SUPPORT:
        return ""
    try:
        document = docx.Document(file)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        st.warning(f"DOCX read error: {e}")
        return ""

def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.warning(f"TXT read error: {e}")
        return ""

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    elif name.endswith(".docx"):
        return read_docx(file)
    elif name.endswith(".txt"):
        return read_txt(file)
    return ""


# -------- TEXT CLEANING --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------- CONTACT EXTRACTION --------
def extract_contact(text):
    email = re.findall(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    phone = re.findall(r'(\+?\d[\d\s\-().]{7,}\d)', text)
    linkedin = re.findall(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    github = re.findall(r'github\.com/[\w\-]+', text, re.IGNORECASE)
    return {
        "email": email[0] if email else None,
        "phone": phone[0].strip() if phone else None,
        "linkedin": linkedin[0] if linkedin else None,
        "github": github[0] if github else None
    }


# -------- EXPERIENCE EXTRACTION --------
def extract_experience_years(text):
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


# -------- EDUCATION EXTRACTION --------
def extract_education(text):
    degrees = ["phd", "ph.d", "doctorate", "masters", "m.tech", "mba", "m.sc",
               "bachelors", "b.tech", "b.e", "b.sc", "b.com", "undergraduate", "graduate"]
    found = [d for d in degrees if d in text.lower()]
    return found[0].upper() if found else None


# -------- DYNAMIC KEYWORD EXTRACTION --------
def extract_keywords_from_jd(jd_text):
    if SPACY_SUPPORT:
        doc = nlp(jd_text)
        keywords = set()
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN"]:
                keywords.add(token.lemma_.lower())
        for ent in doc.ents:
            keywords.add(ent.text.lower())
        return list(keywords)[:40]
    else:
        # Fallback: TF-IDF top terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower())
        stopwords = {"the", "and", "for", "with", "that", "this", "have", "from",
                     "will", "are", "you", "our", "your", "not", "all", "can", "its"}
        return list(set(w for w in words if w not in stopwords))[:40]


# -------- DUPLICATE DETECTION --------
def detect_duplicates(resumes_text):
    if not SKLEARN_SUPPORT or len(resumes_text) < 2:
        return []
    vec = TfidfVectorizer()
    try:
        matrix = vec.fit_transform(resumes_text)
        sim = cosine_similarity(matrix)
        dupes = []
        for i in range(len(sim)):
            for j in range(i+1, len(sim)):
                if sim[i][j] > 0.85:
                    dupes.append((i, j, sim[i][j]))
        return dupes
    except Exception:
        return []


# -------- SCORING ENGINE --------
def score_resume(jd_clean, resume_clean, jd_keywords, jd_raw, resume_raw):
    scores = {}

    # 1. TF-IDF cosine similarity
    if SKLEARN_SUPPORT:
        try:
            vec = TfidfVectorizer()
            matrix = vec.fit_transform([jd_clean, resume_clean])
            tfidf_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except Exception:
            tfidf_score = 0.0
    else:
        tfidf_score = 0.0
    scores["tfidf"] = tfidf_score

    # 2. Keyword match
    matched = [kw for kw in jd_keywords if kw in resume_clean]
    missing = [kw for kw in jd_keywords if kw not in resume_clean]
    kw_score = len(matched) / len(jd_keywords) if jd_keywords else 0
    scores["keyword"] = kw_score
    scores["matched_keywords"] = matched[:15]
    scores["missing_keywords"] = missing[:15]

    # 3. Experience
    jd_exp = extract_experience_years(jd_raw)
    res_exp = extract_experience_years(resume_raw)
    if jd_exp and res_exp:
        exp_score = min(res_exp / jd_exp, 1.0)
    elif res_exp:
        exp_score = 0.8
    else:
        exp_score = 0.5
    scores["experience"] = exp_score
    scores["experience_years"] = res_exp

    # 4. Education
    scores["education"] = extract_education(resume_raw)

    # 5. Contact info
    scores["contact"] = extract_contact(resume_raw)

    # 6. Final weighted score
    final = (tfidf_score * 0.4) + (kw_score * 0.4) + (exp_score * 0.2)
    scores["final"] = round(final * 100, 1)

    return scores


# -------- SCORE BAR HTML --------
def score_bar_html(score_pct, color="#7c83ff"):
    return f"""
    <div class="score-bar-container">
        <div class="score-bar" style="width:{score_pct}%; background: {color};"></div>
    </div>
    """


# -------- MATPLOTLIB CHART --------
def draw_chart(names, scores, threshold):
    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.6)))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    colors = ['#4caf7d' if s >= threshold else '#e05c5c' if s < 40 else '#7c83ff'
              for s in scores]

    bars = ax.barh(names, scores, color=colors, height=0.5, edgecolor='none')

    ax.axvline(x=threshold, color='#e08c3c', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold}%)')
    ax.set_xlim(0, 105)
    ax.set_xlabel('Match Score (%)', color='#8b8fa8', fontsize=10)
    ax.tick_params(colors='#8b8fa8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#252842')
    ax.spines['bottom'].set_color('#252842')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', va='center', ha='left', color='#e0e0e0', fontsize=9)

    ax.legend(facecolor='#1a1d2e', edgecolor='#3a3f6e', labelcolor='#8b8fa8', fontsize=9)
    plt.tight_layout()
    return fig


# -------- MAIN APP --------
init_db()

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 ResumeAI")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Shortlist threshold (%)", 30, 90, 60)
    show_contact = st.toggle("Show contact info", value=True)
    show_keywords = st.toggle("Show keyword breakdown", value=True)
    dark_chart = st.toggle("Show score chart", value=True)

    st.markdown("---")
    st.markdown("### 📂 Multiple JD Mode")
    multi_jd = st.toggle("Enable multi-JD matching", value=False)

    st.markdown("---")
    st.markdown("### 🕘 Session History")
    history = load_history()
    if history:
        for row in history:
            st.markdown(f"""
            <div class="history-item">
                <small style="color:#8b8fa8">{row[1]}</small><br>
                <span style="color:#e0e0e0;font-size:0.9rem">{row[2] or 'Untitled'}</span><br>
                <small style="color:#7c83ff">{row[3]} resumes · Top: {row[4]}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No history yet.")

# Main area
st.markdown("# 🎯 ResumeAI Screener")
st.markdown("<p style='color:#8b8fa8;margin-top:-12px'>Intelligent resume ranking · keyword extraction · candidate insights</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📋 Job Description")
    job_title = st.text_input("Job Title (for history)", placeholder="e.g. Data Scientist")
    jd_text = st.text_area("Paste the full job description here", height=220,
                           placeholder="Looking for a Python developer with 3+ years experience in ML, NLP, and data pipelines...")

    if multi_jd:
        st.markdown("**Additional JDs**")
        jd2 = st.text_area("JD 2 (optional)", height=100, placeholder="Second job description...")
        jd3 = st.text_area("JD 3 (optional)", height=100, placeholder="Third job description...")

with col2:
    st.markdown("### 📁 Resume Files")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) loaded")
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

st.markdown("---")
analyze_btn = st.button("🔍 Analyze Resumes", use_container_width=True)

if analyze_btn:
    if not jd_text.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Extracting text and computing scores..."):

            jd_keywords = extract_keywords_from_jd(jd_text)
            jd_clean = clean_text(jd_text)

            all_jds = [("Primary JD", jd_text, jd_clean)]
            if multi_jd:
                if jd2.strip():
                    all_jds.append(("JD 2", jd2, clean_text(jd2)))
                if jd3.strip():
                    all_jds.append(("JD 3", jd3, clean_text(jd3)))

            results = []
            raw_texts = []

            for file in uploaded_files:
                raw = extract_text(file)
                raw_texts.append(raw)
                if not raw.strip():
                    st.warning(f"⚠️ Could not extract text from {file.name}")
                    continue

                res_clean = clean_text(raw)
                scores = score_resume(jd_clean, res_clean, jd_keywords, jd_text, raw)
                scores["name"] = file.name
                results.append(scores)

            # Duplicate detection
            dupes = detect_duplicates([clean_text(t) for t in raw_texts if t.strip()])
            if dupes:
                for i, j, sim in dupes:
                    st.markdown(f"""
                    <div class="duplicate-warning">
                        ⚠️ <b>Possible duplicate:</b> {uploaded_files[i].name} and {uploaded_files[j].name}
                        are {sim*100:.0f}% similar.
                    </div>
                    """, unsafe_allow_html=True)

            # Sort by final score
            results.sort(key=lambda x: x["final"], reverse=True)

        if not results:
            st.error("No readable resumes found.")
        else:
            # Summary metrics
            st.markdown("## 📊 Results")
            shortlisted = [r for r in results if r["final"] >= threshold]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-card"><h2>{len(results)}</h2><p>Resumes Analyzed</p></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card"><h2>{len(shortlisted)}</h2><p>Shortlisted</p></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card"><h2>{results[0]['final']}%</h2><p>Top Score</p></div>""", unsafe_allow_html=True)
            with m4:
                avg = round(sum(r['final'] for r in results) / len(results), 1)
                st.markdown(f"""<div class="metric-card"><h2>{avg}%</h2><p>Avg Score</p></div>""", unsafe_allow_html=True)

            # Chart
            if dark_chart:
                names = [r["name"].replace(".pdf","").replace(".docx","").replace(".txt","") for r in results]
                scores_list = [r["final"] for r in results]
                fig = draw_chart(names[::-1], scores_list[::-1], threshold)
                st.pyplot(fig)
                plt.close()

            # Candidate cards
            st.markdown("### 🏆 Ranked Candidates")
            for rank, r in enumerate(results, 1):
                status = "shortlisted" if r["final"] >= threshold else "rejected"
                status_label = "✅ Shortlisted" if status == "shortlisted" else "❌ Below Threshold"
                status_color = "#4caf7d" if status == "shortlisted" else "#e05c5c"

                with st.expander(f"#{rank} — {r['name']}  |  {r['final']}%  |  {status_label}"):
                    left, right = st.columns(2)

                    with left:
                        st.markdown("**Score Breakdown**")
                        st.markdown(f"Overall Match: **{r['final']}%**")
                        st.markdown(score_bar_html(r['final'], status_color), unsafe_allow_html=True)

                        tfidf_pct = round(r['tfidf'] * 100, 1)
                        st.markdown(f"Semantic Similarity: **{tfidf_pct}%**")
                        st.markdown(score_bar_html(tfidf_pct, "#7c83ff"), unsafe_allow_html=True)

                        kw_pct = round(r['keyword'] * 100, 1)
                        st.markdown(f"Keyword Coverage: **{kw_pct}%**")
                        st.markdown(score_bar_html(kw_pct, "#a78bfa"), unsafe_allow_html=True)

                        exp_pct = round(r['experience'] * 100, 1)
                        st.markdown(f"Experience Match: **{exp_pct}%**")
                        st.markdown(score_bar_html(exp_pct, "#38bdf8"), unsafe_allow_html=True)

                        if r.get("experience_years"):
                            st.caption(f"🕐 {r['experience_years']} years experience detected")
                        if r.get("education"):
                            st.caption(f"🎓 {r['education']} detected")

                    with right:
                        if show_contact:
                            st.markdown("**Contact Info**")
                            c = r.get("contact", {})
                            st.caption(f"📧 {c.get('email') or 'Not found'}")
                            st.caption(f"📞 {c.get('phone') or 'Not found'}")
                            st.caption(f"💼 {c.get('linkedin') or 'Not found'}")
                            st.caption(f"🐙 {c.get('github') or 'Not found'}")

                        if show_keywords:
                            st.markdown("**Matched Keywords**")
                            matched_html = " ".join([f'<span class="tag matched">{kw}</span>' for kw in r.get("matched_keywords", [])])
                            st.markdown(matched_html or "<i>None</i>", unsafe_allow_html=True)

                            st.markdown("**Missing Keywords**")
                            missing_html = " ".join([f'<span class="tag missing">{kw}</span>' for kw in r.get("missing_keywords", [])[:10]])
                            st.markdown(missing_html or "<i>None</i>", unsafe_allow_html=True)

            # Export
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            export_data = []
            for r in results:
                c = r.get("contact", {})
                export_data.append({
                    "Rank": results.index(r) + 1,
                    "File": r["name"],
                    "Score (%)": r["final"],
                    "Semantic (%)": round(r["tfidf"] * 100, 1),
                    "Keyword (%)": round(r["keyword"] * 100, 1),
                    "Experience (%)": round(r["experience"] * 100, 1),
                    "Exp Years": r.get("experience_years") or "",
                    "Education": r.get("education") or "",
                    "Email": c.get("email") or "",
                    "Phone": c.get("phone") or "",
                    "LinkedIn": c.get("linkedin") or "",
                    "GitHub": c.get("github") or "",
                    "Status": "Shortlisted" if r["final"] >= threshold else "Rejected"
                })

            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV Report", csv, "resume_screening_results.csv", "text/csv")

            # Save to history
            save_session(
                job_title or "Untitled",
                len(results),
                results[0]["name"] if results else "N/A",
                [{"name": r["name"], "score": r["final"]} for r in results]
            )

            st.success("✅ Session saved to history.")