# CV_Project

Un sistem avansat de analiză a carierei IT bazat pe o arhitectură multi-agent, ce combină un model **LLM Fine-Tuned (QLoRA)** pentru extragerea de entități din CV-uri, un motor de căutare hibridă RAG (**HyDE + Qdrant Dense/Sparse + Cross-Encoder Reranker**) orchestrat prin **LangGraph**, și un agent de evaluare a compatibilității tehnice și planificare a dezvoltării profesionale folosind ReAct.

## 🏗️ Arhitectura Sistemului (Multi-Agent Pipeline)

Fluxul complet al aplicației este condus de un orchestrator central și 3 agenți specializați:

1. **Agent 1 (CV Information Extraction & Structuring):**
   * Preia textul brut al unui CV needitat și elimină zgomotul de formatare.
   * Parsează și generează un format strict JSON cu 5 dimensiuni esențiale:
     * `Candidate_Role`
     * `Experience_Level` (Junior / Mid / Senior / Lead)
     * `Years_Of_Experience`
     * `Primary_Skills` (listă de limbaje, framework-uri și tehnologii)
     * `Clean_Summary` (rezumat executiv concis)

2. **Agent 2 (Adaptive Hybrid Job Matching via LangGraph):**
   * **Intent & Role Resolution:** Standardizează rolul vizat fie pe baza preferințelor explicite introduse de utilizator, fie prin fallback pe profilul extras din CV.
   * **HyDE (Hypothetical Document Embeddings):** Sintetizează o fișă de post ideală detaliată pentru a servi drept interogare contextuală bogată.
   * **Hybrid Search (Qdrant):** Combină căutarea densă (`text-embedding-3-small`, 1536 dim) cu vectori rari BM25 (`FastEmbed`) prin algoritmul Reciprocal Rank Fusion (RRF).
   * **Neural Reranking:** Re-evaluează candidații cu un model local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
   * **Self-Correction & Web Fallback:** Dacă scorul de relevanță scade sub un prag prestabilit, agentul formulează un diagnostic tehnic pentru regenerarea promptului HyDE sau comută automat pe căutare pe web prin Tavily.

3. **Agent 3 (Technical Gap Analysis & Career Roadmap):**
   * Compară profilul structurat al candidatului cu cerințele joburilor selectate.
   * Calculează scorul de potrivire tehnică, evidențiază competențele lipsă și elaborează un plan de învățare structurat pe etape.

## 📊 Pipeline-ul de Date & Curare (CV Dataset)

### 1. Sursa Datelor & Preprocesare Brută 
* Dataset-ul inițial de CV -uri(https://www.kaggle.com/datasets/rayyankauchali0/resume-dataset) a fost redus doar la câmpurile Category și Text.
* Eliminarea caracterelor de formatare și escape:** Secvențele `\n`, `\\n`, `\t` și `\\t` au fost convertite în spații albe uniforme.
* Filtrare regex pentru zgomot de parsare:** Au fost înlăturate caracterele speciale nestandardizate (`re.sub(r"[^\w\s.,;:/\-+#@()&]", "", text)`), păstrând exclusiv setul alfanumeric, spațiile și semnele uzuale de punctuație sau sintaxă tehnică.
* Ștergerea separatoarelor decorative:** Au fost eliminate secvențele repetitive de delimitare vizuală prezente în text (`==`, `--`, `__` repetate de două sau mai multe ori).
* Normalizarea spațierii:** Spațiile albe multiple consecutive au fost condensate într-un singur spațiu.
* Prag minim de lungime (`min_chars = 300`):** Înregistrările cu mai puțin de 300 de caractere au fost eliminate automat ca fiind intrări corupte sau fără istoric profesional util.
* Trunchiere superioară (`max_chars = 5000`):** CV-urile foarte lungi au fost tăiate la maximum 5.000 de caractere pentru a optimiza consumul de memorie GPU și a preveni depășirea ferestrei de context la antrenarea QLoRA.

### 2. Generarea Etichetelor de Antrenare (OpenAI Batch API)
* CV-urile curate au fost împărțite în calupuri gestionate prin `batch_processor.py` și trimise asincron către OpenAI Batch API (`gpt-4o-mini`).
* Sistemul a monitorizat starea execuțiilor la distanță, descărcând automat răspunsurile structurate.

### 3. Formatarea pentru SFT & Publicarea pe Hugging Face Hub
* Conversațiile au fost structurate în format de Supervised Fine-Tuning (`messages`: `system`, `user`, `assistant`).
* Setul de date rezultat a fost divizat stratificat în `train`, `validation` și `test` și publicat pe Hugging Face:
  * **Dataset:** `alecs-vlad24/cv-resume-structuring-v2pro`
