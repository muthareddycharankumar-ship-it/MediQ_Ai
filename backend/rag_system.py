import os
import re
import math
import torch
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.device("cpu")

DOC_DIR = "/home/safepro/Downloads/Rheumatology_pdf"
CHROMA_DIR = "/home/safepro/Desktop/opencv/MediQ_Ai/chroma_db"
LLM_MODEL = "mediq-rheumatology"
EMBED_MODEL = "nomic-embed-text"

llm = Ollama(
    model=LLM_MODEL,
    temperature=0.1,
    num_predict=2000,
    num_ctx=3072,
    repeat_penalty=1.1,
    num_thread=8,
)

def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("Creating ChromaDB...")
        loader = DirectoryLoader(DOC_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        docs = splitter.split_documents(documents)
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    else:
        print("Loading existing ChromaDB...")
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectordb

vectorstore = load_vector_db()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=256)

def retrieve_documents(question):
    docs = vectorstore.similarity_search(question, k=10)
    if not docs:
        return []
    pairs = [[question, doc.page_content[:300]] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:4]]

def fuzzy_match(text, candidates, threshold=80):
    for candidate in candidates:
        if fuzz.ratio(text, candidate) >= threshold:
            return True
    return False

def correct_typos(text):
    corrections = {
        "hlep": "help", "hepl": "help", "hlp": "help",
        "helllo": "hello", "helo": "hello", "hii": "hi", "heyy": "hey",
        "thnaks": "thanks", "thnak": "thank", "thanku": "thank you",
        "thnk": "thank", "byee": "bye", "waht": "what", "wht": "what",
        "hw": "how", "cn": "can", "yu": "you", "ur": "your",
        "wid": "with", "plz": "please", "pls": "please",
        "assit": "assist", "asist": "assist",
    }
    words = text.split()
    return " ".join(corrections.get(w, w) if len(w) <= 8 else w for w in words)

BLOCKED_KEYWORDS = []

def is_blocked_topic(question):
    if not is_medical_question(question.lower()):
        return True
    return False



def is_medical_question(text):
    """Returns True only if question contains actual medical/rheumatology terms."""
    text = text.lower()

    # These are purely medical/rheumatology terms
    medical_terms = [
        # Diseases
        "arthritis", "lupus", "gout", "sle", "fibromyalgia", "vasculitis",
        "sjogren", "spondylitis", "myositis", "scleroderma", "osteoporosis",
        "osteoarthritis", "psoriatic", "rheumatoid", "ankylosing", "raynaud",
        "polymyalgia", "behcet", "sarcoidosis", "myopathy", "autoimmune",
        "uveitis", "iritis", "episcleritis", "scleritis",
        "tendinitis", "bursitis", "enthesitis", "synovitis", "tenosynovitis",
        "vasculitis", "neuropathy", "myopathy", "nephritis", "pleuritis",
        "pericarditis", "serositis", "amyloidosis", "raynaud",
        # Symptoms
        "joint pain", "joint swelling", "morning stiffness", "back pain",
        "neck pain", "muscle pain", "muscle weakness", "fatigue",
        "photosensitivity", "malar rash", "butterfly rash", "purpura",
        "raynaud", "dry eyes", "dry mouth", "hair loss", "oral ulcer",
        "dactylitis", "enthesitis", "sacroiliitis",
        # Medical procedures/tests
        "synovial fluid", "arthrocentesis", "joint aspiration",
        "anti-ccp", "rheumatoid factor", "ana", "anca", "anti-dsdna",
        "hla-b27", "esr", "crp", "uric acid", "complement",
        "x-ray", "mri", "ultrasound", "ct scan", "biopsy",
        "das28", "cdai", "sdai", "basdai", "basfi", "sledai", "jadas",
        # Drugs
        "methotrexate", "hydroxychloroquine", "sulfasalazine", "leflunomide",
        "etanercept", "adalimumab", "infliximab", "rituximab", "tocilizumab",
        "abatacept", "baricitinib", "tofacitinib", "upadacitinib",
        "allopurinol", "febuxostat", "colchicine", "probenecid",
        "prednisone", "prednisolone", "methylprednisolone",
        "nsaid", "ibuprofen", "naproxen", "indomethacin", "celecoxib",
        "metformin", "aspirin", "paracetamol", "statin",
        "dmard", "biologic", "jak inhibitor", "tnf inhibitor",
        # Body parts in medical context
        "sacroiliac", "metatarsophalangeal", "metacarpophalangeal",
        "interphalangeal", "temporomandibular",
        # General medical
        "diagnosis", "treatment", "prognosis", "pathogenesis", "etiology",
        "inflammation", "autoantibody", "immunosuppression",
        "remission", "flare", "disease activity", "comorbidity",
        "birefringent", "crystal", "toph",
        # Single word symptoms — IMPORTANT
        "pain", "stiffness", "swelling", "aching", "tender", "tenderness",
        "redness", "warmth", "swollen", "inflamed", "inflammation",
        "weakness", "numbness", "tingling", "burning", "throbbing",
        "cramp", "spasm", "stiff", "sore", "soreness",
        "immobile", "immobility", "limited motion", "restricted",
        "limping", "walking difficulty", "difficulty walking",
        "relief", "worse", "better", "aggravated", "morning",
        "night pain", "rest pain", "weight bearing",
        "fever", "rash", "sweat", "chills", "weight loss",
        "leg pain", "knee pain", "hip pain", "shoulder pain",
        "wrist pain", "ankle pain", "foot pain", "heel pain",
        "spine", "lower back", "upper back", "neck", "joint", "nodule", "erosion",
        "macrophage", "cytokine", "interleukin", "tnf", "il-6",
        "renal", "hepatic", "pulmonary", "cardiac", "neurological",
    ]

    return any(term in text for term in medical_terms)

GREETING_LIST = ["hi", "hello", "hey", "howdy", "good morning", "good evening", "good afternoon", "good night", "hi there", "hello there", "hey there"]
FAREWELL_LIST = ["ok thanks", "okay thanks", "thank you", "thanks", "ok thank you", "thx", "ty", "thank u", "bye", "goodbye", "see you", "take care", "that's all", "done", "ok bye"]
HELP_LIST = ["what can you do", "what can you help", "how can you help", "who are you", "what are you", "introduce yourself", "tell me about yourself", "what do you do", "can you help me", "help me", "i need help", "what topics", "how does this work", "what is mediq", "your capabilities", "assist me", "help with me"]

def classify_intent(question):
    q = question.lower().strip()
    q = q.replace("?", "").replace("!", "").replace(".", "").replace("/", "")
    q = " ".join(q.split())
    q_corrected = correct_typos(q)

    if q_corrected.strip() in ["1", "2", "3", "4"]:
        return "number_select"

    if is_medical_question(q_corrected):
        return "question"

    if fuzzy_match(q_corrected, GREETING_LIST, threshold=85):
        return "greeting"

    if fuzzy_match(q_corrected, FAREWELL_LIST, threshold=85):
        return "farewell"

    for phrase in HELP_LIST:
        if fuzz.ratio(phrase, q_corrected) >= 80:
            return "help"
    if not is_medical_question(q_corrected):
        return "blocked"
        return "blocked"
    if fuzzy_match(first_word, ["hi", "hello", "hey", "helo", "hii"], threshold=85):
        return "greeting"

    return "question"

def classify_das28(score):
    if score < 2.6:
        return "Remission", "🟢", "Target achieved. Continue current treatment and monitor regularly."
    elif score <= 3.2:
        return "Low Disease Activity", "🟡", "Good control. Continue current therapy. Aim for remission if possible."
    elif score <= 5.1:
        return "Moderate Disease Activity", "🟠", "Consider treatment escalation. Review and optimize DMARD therapy."
    else:
        return "High Disease Activity", "🔴", "Immediate treatment escalation required. Consider biologic or JAK inhibitor therapy."

def get_next_steps(question):
    q = question.lower()
    if any(w in q for w in ["treatment", "treat", "therapy", "drug", "medication", "manage"]):
        return ["Side effects of these medications", "How to monitor treatment response", "What if treatment fails?", "Non-drug management options"]
    elif any(w in q for w in ["symptom", "sign", "feature", "manifestation"]):
        return ["How is this condition diagnosed?", "What are the treatment options?", "Disease progression over time", "When to see a specialist"]
    elif any(w in q for w in ["diagnos", "criteria", "test", "imaging", "lab"]):
        return ["Treatment options after diagnosis", "How to monitor disease activity", "Interpreting lab results", "Prognosis"]
    elif any(w in q for w in ["das28", "sdai", "cdai", "basdai", "score", "formula", "calculate"]):
        return ["How to use this score for treatment decisions", "Comparison with other disease activity scores", "Target score for remission", "Monitoring frequency"]
    elif any(w in q for w in ["prognos", "outcome", "indicate", "significance"]):
        return ["Treatment options for this condition", "How to slow disease progression", "Monitoring disease activity", "Managing complications"]
    else:
        return ["Learn more about this condition", "Treatment options available", "Diagnostic workup", "Long-term prognosis"]

last_context = {"topic": ""}

MEDICAL_FACTS = """
=== DISEASE ACTIVITY SCORES ===
DAS28-ESR = 0.56*sqrt(TJC28) + 0.28*sqrt(SJC28) + 0.70*ln(ESR) + 0.014*GH
DAS28-CRP = 0.56*sqrt(TJC28) + 0.28*sqrt(SJC28) + 0.36*ln(CRP+1) + 0.014*GH + 0.96
DAS28 cutoffs: Remission <2.6, Low 2.6-3.2, Moderate 3.2-5.1, High >5.1
DAS28 limitation: excludes feet/ankle joints
DAS28-CRP gives slightly lower scores than DAS28-ESR

SDAI = TJC28 + SJC28 + PGA + EGA + CRP(mg/dL) — REQUIRES CRP lab value
SDAI cutoffs: Remission <=3.3, Low <=11, Moderate <=26, High >26

CDAI = TJC28 + SJC28 + PGA + EGA — NO laboratory values needed
CDAI cutoffs: Remission <=2.8, Low <=10, Moderate <=22, High >22
CDAI is purely clinical, can be done at point of care instantly
CDAI does NOT use ESR, CRP or any blood test

BASDAI = (Q1+Q2+Q3+Q4+((Q5+Q6)/2))/5 — 6 questions 0-10
BASDAI >=4 = active AS needing treatment change
BASFI = average of 10 functional questions 0-10

SLEDAI-2K: weighted sum of 24 features, higher = more active SLE
PASI: mild <10, moderate 10-20, severe >20
ACR20/50/70: % improvement in joint counts and other measures
EULAR response: Good = DAS28 decrease >1.2 AND current DAS28 <3.2

=== CRYSTAL ARTHROPATHY ===
Needle-shaped NEGATIVELY birefringent = MSU crystals = GOUT
Rhomboid WEAKLY POSITIVELY birefringent = Calcium Pyrophosphate = CPPD
Chalky deposits ear/elbow = Tophi = Chronic Tophaceous Gout
Uric acid target: <6 mg/dL most, <5 mg/dL tophaceous gout

=== KEY CLINICAL FACTS ===
Anti-CCP positive = worse prognosis, more erosive RA, higher joint damage
RF positive = extra-articular features, worse prognosis
Methotrexate + breathlessness = Methotrexate pneumonitis, STOP MTX
HLA-B27 = Ankylosing Spondylitis (80-90% AS patients positive)
Morning stiffness >1 hour = inflammatory arthritis
ESR normal in OA = OA is non-inflammatory
First MTP sudden pain = Gout
DAS28 remission = <2.6 (NOT 2.4)
CDAI remission = <=2.8, SDAI remission = <=3.3

=== DRUGS ===
Methotrexate: anchor DMARD RA, 7.5-25mg weekly, give folic acid always
Hydroxychloroquine: 200-400mg/day, yearly eye check
Allopurinol: xanthine oxidase inhibitor, start 100mg increase slowly
Colchicine: acute gout 0.5mg 2-3x daily
Biologics: TNF (etanercept/adalimumab/infliximab), IL-6 (tocilizumab), CD20 (rituximab)
JAK inhibitors: baricitinib, tofacitinib, upadacitinib
GFR <30: avoid NSAIDs, reduce colchicine, stop methotrexate
"""

def ask_rag_stream(question):
    global last_context
    q = question.lower().strip()
    intent = classify_intent(question)

    if intent == "blocked" or (intent not in ["greeting", "farewell", "help", "number_select"] and is_blocked_topic(question)):
        yield "I'm sorry, that topic is outside my area of expertise. 🩺\n\nI specialize in **Rheumatology and Medicine**. Please ask a medical question and I'll be happy to help! 😊"
        return

    if intent == "greeting":
        yield "Hello! 👋 I'm **MedIQ**,How can I help you today?"
        return

    if intent == "help":
        yield """Hello! 👋 I'm **MedIQ**, a specialized AI assistant for **Rheumatology**.

Here's what I can help you with:

🔹 **Disease Information** — Rheumatoid Arthritis, Lupus, Gout, Ankylosing Spondylitis, etc.
🔹 **Symptoms** — Signs and clinical features of rheumatic diseases
🔹 **Diagnosis** — Diagnostic criteria, lab tests, imaging studies
🔹 **Treatment** — DMARDs, biologics, JAK inhibitors, NSAIDs
🔹 **Disease Causes** — Etiology, pathogenesis, risk factors
🔹 **Prognosis** — Long-term outlook and complications
🔹 **Monitoring** — Disease activity tracking and treatment response
🔹 **Clinical Cases** — Analyze patient presentations
🔹 **Drug Information** — Mechanisms, dosages, side effects
🔹 **Disease Activity Scores** — DAS28, SDAI, CDAI, BASDAI, SLEDAI calculations
🔹 **Research & Guidelines** — ACR/EULAR guidelines

💡 **Try asking:**
- *"What is the first-line treatment for Rheumatoid Arthritis?"*
- *"Calculate DAS28 with TJC=6, SJC=5, ESR=40, GH=60"*
- *"Does CDAI require laboratory values?"*
- *"Difference between DAS28-ESR and DAS28-CRP?"*

What would you like to know? 😊"""
        return

    if intent == "farewell":
        yield "You're welcome! 😊 Feel free to ask anytime. Take care! 👋"
        return

    if intent == "number_select":
        next_steps = get_next_steps(last_context.get("topic", ""))
        idx = int(q) - 1
        if 0 <= idx < len(next_steps):
            new_question = f"{next_steps[idx]} {last_context['topic']}".strip()
            yield f"Great choice! Let me explain: **{next_steps[idx]}**\n\n"
            yield from ask_rag_stream(new_question)
        else:
            yield "Please type a number between 1 and 4, or ask your next question directly."
        return

    if is_blocked_topic(question):
        yield "I'm sorry, that topic is outside my area of expertise. 🩺\n\nI specialize in **Rheumatology and Medicine**. Please ask a medical question and I'll be happy to help! 😊"
        return

    # ---- DAS28 CALCULATOR ----
    calc_match = re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?esr\s*[=:]\s*(\d+\.?\d*).*?gh\s*[=:]\s*(\d+\.?\d*)', q)
    crp_match = re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?crp\s*[=:]\s*(\d+\.?\d*).*?gh\s*[=:]\s*(\d+\.?\d*)', q)

    if calc_match or crp_match:
        use_crp = crp_match is not None and calc_match is None
        m = crp_match if use_crp else calc_match
        tjc, sjc, marker, gh = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))

        if use_crp:
            das28 = round(0.56*math.sqrt(tjc) + 0.28*math.sqrt(sjc) + 0.36*math.log(marker+1) + 0.014*gh + 0.96, 2)
            s3 = round(0.36*math.log(marker+1), 3)
            fname, mlabel, s3label = "DAS28-CRP", f"CRP = {marker} mg/L", f"0.36*ln({marker}+1)"
        else:
            das28 = round(0.56*math.sqrt(tjc) + 0.28*math.sqrt(sjc) + 0.70*math.log(marker) + 0.014*gh, 2)
            s3 = round(0.70*math.log(marker), 3)
            fname, mlabel, s3label = "DAS28-ESR", f"ESR = {marker} mm/hr", f"0.70*ln({marker})"

        s1 = round(0.56*math.sqrt(tjc), 3)
        s2 = round(0.28*math.sqrt(sjc), 3)
        s4 = round(0.014*gh, 3)
        category, emoji, rec = classify_das28(das28)

        result = f"## {fname} Calculation\n\n"
        result += f"**Given:** TJC={tjc}, SJC={sjc}, {mlabel}, GH={gh}\n\n"
        result += f"## Step-by-Step\n\n"
        result += f"| Step | Result |\n|---|---|\n"
        result += f"| 0.56 × √{tjc} | {s1} |\n"
        result += f"| 0.28 × √{sjc} | {s2} |\n"
        result += f"| {s3label} | {s3} |\n"
        result += f"| 0.014 × {gh} | {s4} |\n"
        result += f"| **Total** | **{das28}** |\n\n"
        result += f"## {emoji} Result: {category} (Score = {das28})\n\n"
        result += f"{rec}\n\n"
        result += f"| Category | Score |\n|---|---|\n"
        result += f"| Remission | < 2.6 |\n| Low | 2.6–3.2 |\n| Moderate | 3.2–5.1 |\n| High | > 5.1 |"
        yield result

        next_steps = get_next_steps(question)
        yield "\n\n---\n**🔍 Want to explore further?**\n\n" + "\n".join([f"**{i+1}.** {s}" for i,s in enumerate(next_steps)]) + "\n\n💬 Type **1-4** or ask next question!"
        return

    # ---- CALCULATORS ----
    import re as _re

    cdai_m = _re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?(?:ptga|pga)\s*[=:]\s*(\d+\.?\d*).*?(?:phga|ega)\s*[=:]\s*(\d+\.?\d*)', q)
    if cdai_m and "cdai" in q:
        tjc  = float(cdai_m.group(1).strip(".,"))
        sjc  = float(cdai_m.group(2).strip(".,"))
        pga  = float(cdai_m.group(3).strip(".,"))
        ega  = float(cdai_m.group(4).strip(".,"))
        score = round(tjc + sjc + pga + ega, 1)
        if score <= 2.8:
            cat,emoji,rec = "Remission","🟢","Target achieved. Continue treatment."
        elif score <= 10:
            cat,emoji,rec = "Low Disease Activity","🟡","Good control. Continue therapy."
        elif score <= 22:
            cat,emoji,rec = "Moderate Disease Activity","🟠","Consider treatment escalation."
        else:
            cat,emoji,rec = "High Disease Activity","🔴","Immediate escalation required."
        yield "## CDAI Calculation\n\n"
        yield "**Formula: CDAI = TJC + SJC + PGA + EGA**\n\n"
        yield "> No laboratory values needed — purely clinical score\n\n"
        yield f"| Component | Value |\n|---|---|\n| TJC28 | {tjc} |\n| SJC28 | {sjc} |\n| PGA | {pga} |\n| EGA | {ega} |\n| **Total** | **{score}** |\n\n"
        yield f"## {emoji} Result: **{cat}** (Score = {score})\n\n{rec}\n\n"
        yield "| Category | Score |\n|---|---|\n| Remission | ≤ 2.8 |\n| Low | ≤ 10 |\n| Moderate | ≤ 22 |\n| High | > 22 |"
        next_steps = get_next_steps(question)
        yield "\n\n---\n**🔍 Want to explore further?**\n\n" + "\n".join([f"**{i+1}.** {s}" for i,s in enumerate(next_steps)]) + "\n\n💬 Type **1-4** or ask next question!"
        return

    sdai_m = _re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?(?:ptga|pga)\s*[=:]\s*(\d+\.?\d*).*?(?:phga|ega)\s*[=:]\s*(\d+\.?\d*).*?crp\s*[=:]\s*(\d+\.?\d*)', q)
    if sdai_m and "sdai" in q:
        tjc  = float(sdai_m.group(1).strip(".,"))
        sjc  = float(sdai_m.group(2).strip(".,"))
        pga  = float(sdai_m.group(3).strip(".,"))
        ega  = float(sdai_m.group(4).strip(".,"))
        crp  = float(sdai_m.group(5).strip(".,"))
        score = round(tjc + sjc + pga + ega + crp, 1)
        if score <= 3.3:
            cat,emoji,rec = "Remission","🟢","Target achieved."
        elif score <= 11:
            cat,emoji,rec = "Low Disease Activity","🟡","Good control."
        elif score <= 26:
            cat,emoji,rec = "Moderate Disease Activity","🟠","Consider escalation."
        else:
            cat,emoji,rec = "High Disease Activity","🔴","Immediate escalation."
        yield "## SDAI Calculation\n\n"
        yield "**Formula: SDAI = TJC + SJC + PGA + EGA + CRP(mg/dL)**\n\n"
        yield f"| Component | Value |\n|---|---|\n| TJC28 | {tjc} |\n| SJC28 | {sjc} |\n| PGA | {pga} |\n| EGA | {ega} |\n| CRP | {crp} |\n| **Total** | **{score}** |\n\n"
        yield f"## {emoji} Result: **{cat}** (Score = {score})\n\n{rec}\n\n"
        yield "| Category | Score |\n|---|---|\n| Remission | ≤ 3.3 |\n| Low | ≤ 11 |\n| Moderate | ≤ 26 |\n| High | > 26 |"
        next_steps = get_next_steps(question)
        yield "\n\n---\n**🔍 Want to explore further?**\n\n" + "\n".join([f"**{i+1}.** {s}" for i,s in enumerate(next_steps)]) + "\n\n💬 Type **1-4** or ask next question!"
        return

    esr_m = _re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?esr\s*[=:]\s*(\d+\.?\d*).*?gh\s*[=:]\s*(\d+\.?\d*)', q)
    crp_m = _re.search(r'tjc\s*[=:]\s*(\d+\.?\d*).*?sjc\s*[=:]\s*(\d+\.?\d*).*?crp\s*[=:]\s*(\d+\.?\d*).*?gh\s*[=:]\s*(\d+\.?\d*)', q)
    if esr_m or crp_m:
        use_crp = crp_m is not None and esr_m is None
        m = crp_m if use_crp else esr_m
        tjc   = float(m.group(1).strip(".,"))
        sjc   = float(m.group(2).strip(".,"))
        marker= float(m.group(3).strip(".,"))
        gh    = float(m.group(4).strip(".,"))
        if use_crp:
            score = round(0.56*math.sqrt(tjc)+0.28*math.sqrt(sjc)+0.36*math.log(marker+1)+0.014*gh+0.96,2)
            s3    = round(0.36*math.log(marker+1),3)
            fname,mlabel,s3l = "DAS28-CRP",f"CRP={marker}mg/L",f"0.36*ln({marker}+1)"
        else:
            score = round(0.56*math.sqrt(tjc)+0.28*math.sqrt(sjc)+0.70*math.log(marker)+0.014*gh,2)
            s3    = round(0.70*math.log(marker),3)
            fname,mlabel,s3l = "DAS28-ESR",f"ESR={marker}mm/hr",f"0.70*ln({marker})"
        s1=round(0.56*math.sqrt(tjc),3)
        s2=round(0.28*math.sqrt(sjc),3)
        s4=round(0.014*gh,3)
        if score < 2.6:
            cat,emoji,rec = "Remission","🟢","Target achieved."
        elif score <= 3.2:
            cat,emoji,rec = "Low Disease Activity","🟡","Good control."
        elif score <= 5.1:
            cat,emoji,rec = "Moderate Disease Activity","🟠","Consider escalation."
        else:
            cat,emoji,rec = "High Disease Activity","🔴","Immediate escalation."
        yield f"## {fname} Calculation\n\n"
        yield f"**Given:** TJC={tjc}, SJC={sjc}, {mlabel}, GH={gh}\n\n"
        yield f"| Step | Result |\n|---|---|\n| 0.56×√{tjc} | {s1} |\n| 0.28×√{sjc} | {s2} |\n| {s3l} | {s3} |\n| 0.014×{gh} | {s4} |\n| **Total** | **{score}** |\n\n"
        yield f"## {emoji} Result: **{cat}** (Score = {score})\n\n{rec}\n\n"
        yield "| Category | Score |\n|---|---|\n| Remission | < 2.6 |\n| Low | 2.6–3.2 |\n| Moderate | 3.2–5.1 |\n| High | > 5.1 |"
        next_steps = get_next_steps(question)
        yield "\n\n---\n**🔍 Want to explore further?**\n\n" + "\n".join([f"**{i+1}.** {s}" for i,s in enumerate(next_steps)]) + "\n\n💬 Type **1-4** or ask next question!"
        return

    # ---- LLM FOR ALL OTHER QUESTIONS ----
    docs = retrieve_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    last_context["topic"] = question

    prompt = f"""You are MedIQ, a senior rheumatology expert AI assistant.

VERIFIED MEDICAL FACTS — always follow these, never contradict:
CALCULATION RULES — when asked to calculate any score, do exact arithmetic step by step:
- CDAI = TJC + SJC + PGA + EGA (simple addition, no lab needed)
- SDAI = TJC + SJC + PGA + EGA + CRP
- DAS28-ESR = 0.56*sqrt(TJC) + 0.28*sqrt(SJC) + 0.70*ln(ESR) + 0.014*GH
- DAS28-CRP = 0.56*sqrt(TJC) + 0.28*sqrt(SJC) + 0.36*ln(CRP+1) + 0.014*GH + 0.96
- Always show step by step table
- Always show correct cutoffs after result
- Never invent a different formula
{MEDICAL_FACTS}

TASK: Answer the question below accurately and completely.

STRICT HEADING RULES:
Step 1 — Identify question type by reading the FULL question carefully:
IMPORTANT: Never use SCORE template unless the question explicitly mentions a scoring system name like DAS28, CDAI, SDAI, BASDAI, SLEDAI, PASI, JADAS.
Never use CLINICAL_CASE template unless the question contains a patient age and specific symptoms together.

Step 1 — Identify question type from this list:
  - YES_NO: question starts with does/is/can/do/will/should/would
  - FACTUAL: what is/are/were, define, explain, describe
  - COMPARISON: difference/compare/versus/vs/between
  - TREATMENT: treat/therapy/manage/first line/drug/medication
  - SYMPTOMS: symptom/sign/feature/manifestation/present
  - DIAGNOSIS: diagnose/criteria/test/imaging/investigation
  - PROGNOSIS: prognosis/outcome/indicate prognostically/significance
  - CLINICAL_CASE: patient/year old/presents with/case/history
  - FINDING: shows/found/result/indicates/suggests
  - CRYSTAL: crystal/birefringent/deposit/toph
  - SCORE: score/formula/calculate/index/das28/cdai/sdai/basdai
  - LIMITATION: limited/limitation/disadvantage/weakness/exclude

Step 2 — Use ONLY the matching headings:
  YES_NO: ## Direct Answer | ## Why This Is So | ## Key Facts | ## Comparison With Similar | ## Clinical Importance
  FACTUAL: ## Direct Answer | ## Detailed Explanation | ## Key Components | ## Clinical Relevance | ## Important Points
  COMPARISON: ## Direct Answer | ## Side-by-Side Comparison | ## Key Differences | ## When to Use Which | ## Clinical Implication
  TREATMENT: ## Direct Answer | ## Goals of Treatment | ## First-Line Options | ## How They Work | ## Side Effects | ## When to Escalate
  SYMPTOMS: ## Direct Answer | ## Early Symptoms | ## Progressive Features | ## Systemic Features | ## Red Flags | ## When to Seek Help
  DIAGNOSIS: ## Direct Answer | ## Clinical Findings | ## Laboratory Tests | ## Imaging | ## Diagnostic Criteria | ## Differential Diagnosis
  PROGNOSIS: ## Direct Answer | ## Prognostic Significance | ## Clinical Impact | ## Effect on Disease Course | ## Monitoring Implications
  CLINICAL_CASE: ## Direct Answer | ## Clinical Assessment | ## Most Likely Diagnosis | ## Investigations | ## Treatment Plan | ## Patient Counseling
  FINDING: ## Direct Answer | ## What This Indicates | ## Associated Condition | ## Clinical Significance | ## Management
  CRYSTAL: ## Direct Answer | ## Crystal Type | ## Associated Disease | ## Identification | ## Treatment
  SCORE: ## Direct Answer | ## Formula | ## Components | ## Score Interpretation | ## Clinical Use | ## Limitations
  LIMITATION: ## Direct Answer | ## What Is Excluded | ## Why It Matters | ## Clinical Impact | ## Better Alternatives

CONTENT RULES:
- ## Direct Answer: 2-3 sentences, exact correct answer immediately
- Every other heading: 3-5 sentences, medically accurate
- Use tables and bullet points where helpful
- Include drug names, doses where relevant
- Never contradict the VERIFIED MEDICAL FACTS above

Context from medical textbooks:
{context}



Question: {question}

Answer:"""

    for chunk in llm.stream(prompt):
        yield chunk

    if docs:
        citations = list(set([f"📄 {os.path.basename(doc.metadata.get('source',''))} (page {doc.metadata.get('page','N/A')})" for doc in docs]))
        yield "\n\n---\n**📚 Sources:**\n" + "\n".join(citations)

    next_steps = get_next_steps(question)
    yield "\n\n---\n**🔍 Want to explore further?**\n\n" + "\n".join([f"**{i+1}.** {s}" for i,s in enumerate(next_steps)]) + "\n\n💬 Type **1-4** or ask next question!"