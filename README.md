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
   * **Orchestrare cu LangGraph & LangChain:** Întreg fluxul este modelat ca un `StateGraph`:
     * `cv_text` & `user_target_text`: Intrările brute ale candidatului.
     * `resolved_role`: Rolul standardizat extras din intenție sau din profilul CV.
     * `hyde_document`: Documentul ipotetic de post sintetizat dinamic.
     * `candidate_jobs` & `top_jobs`: Rezultatele extrase din Qdrant și re-ierarhizate.
     * `best_score`: Scorul Cross-Encoder al celei mai bune potriviri.
     * `retry_count` & `feedback`: Mecanismul de memorie pentru corecția iterativă.
     * `source`: Indicatorul sursei finale a fișelor (`"qdrant"` sau `"web"`).

   * **Validare Strictă cu Pydantic:** Pentru a elimina răspunsurile nestructurate sau erorile de parsare:
     * **Extracția intenției de rol:** Modelează ieșirea prin schema `RoleExtractionResponse` (`is_specific_request: bool`, `extracted_role: Optional[str]`), diferențiind intențiile specifice de cererile vagi sau deschise.
     * **Sinteza fișelor de post:** Modelează ieșirea prin `WebSearchJobExtraction` și `JobProfileSchema`, forțând extragerea garantată a 3 profiluri cu atribuții tehnice clare și competențe concrete.

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

# De ce am folosit OPENAI Batch API?
Dataset-ul brut conținea exclusiv text needitat de CV. Pentru a antrena un model open-source local să structureze datele determinist, aveam nevoie de o bază de antrenament formată din perechi de tipul:  
`Text Brut (Prompt)` ➡️️ `JSON Structurat Ideal (Ground Truth)`

* **Sintetizare de date la standard înalt:** `gpt-4o-mini` a fost ghidat printr-un prompt de sistem riguros să extragă cele 5 dimensiuni cheie: `Candidate_Role`, `Experience_Level`, `Years_Of_Experience`, `Primary_Skills` și `Clean_Summary`.
* **Calcul temporal precis:** Promptul a forțat calculul matematic al vechimii luând anul **2026** ca punct de referință pentru rolurile active (*"Present"*), eliminând aproximările ambigue.
* **Eficiență de cost și rată de procesare:** Folosind **OpenAI Batch API** (asincron, cu fereastră de livrare de 24h), procesarea a beneficiat de un **discount de 50% din costul API** și a eliminat complet erorile de rate-limiting (TPM/RPM) care ar fi apărut la apeluri clasice sincron

# Antrenarea prin QLoRA (Quantized Low-Rank Adaptation)

Antrenarea modelului a fost realizată în Google Colab (GPU Tesla T4) folosind librăria **Unsloth** pentru accelerare computațională și optimizarea memoriei:

* **Model de Bază:** `unsloth/Llama-3.2-3B-Instruct`.
* **Kvantizare pe 4-biți (QLoRA):** Modelul a fost încărcat în memorie pe 4-biți (`bitsandbytes`), permițând rularea fără probleme pe o singură placă video de 16 GB VRAM.
* **Adaptori LoRA (PEFT):** Greutățile de bază au fost înghețate, aplicându-se adaptori de rang redus pe toate modulele de proiecție și atenție (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`):
  * $r = 16$ (Rank-ul LoRA)
  * $\alpha = 16$ (Scaling factor)
  * **Parametri antrenați:** Doar **24.313.856 parametri** dintr-un total de ~3,23 miliarde (**0,75%** din parametrii întregului model).
* **Configurația de antrenament:**
  * **Lungimea secvenței (`max_seq_length`):** 4.096 tokeni.
  * **Batch size:** 2 per dispozitiv cu acumulare de gradient de 4 pași (batch efectiv de 8).
  * **Dataset split:** 2.798 exemple antrenare, 350 validare, 350 test.
  * **Epoci & Pași:** 1 epocă, totalizând exact **350 global steps**.
  * **Optimizator & Scheduler:** `adamw_8bit`, learning rate `2e-4`, scheduler linear cu warmup.
* **Artifact publicat:** Adaptorul rezultat a fost salvat și publicat pe Hugging Face:  
  [`alecs-vlad24/cv-structuring-llm-v2-2026-08-12_10.49`](https://huggingface.co/alecs-vlad24/cv-structuring-llm-v2-2026-08-12_10.49).

---

### Monitorizarea Curbei de Învățare (Weights & Biases)

Antrenamentul a fost monitorizat complet în **Weights & Biases (W&B)**:

* **Evoluția `train/loss`:** A pornit de la o valoare inițială ridicată de **~2.84** (pasul 10), coborând rapid sub 1.20 până la pasul 50 și stabilizându-se într-un interval optim de convergență între **0.90 și 1.05**, atingând valoarea finală de **0.8968** la pasul 350.
* **Evaluarea `eval/loss`:** Deoarece parametrul de evaluare a fost setat pe `eval_strategy = "epoch"` pentru o singură epocă de antrenament, calculul metricilor pe setul de validare s-a executat o singură dată (la pasul 350), înregistrând un **validation loss de 1.0001**.

---

### Evaluarea Modelului pe Setul de Test (350 CV-uri)

Performanța finală a modelului a fost calculată prin inferență directă pe cele **350 de CV-uri din setul de testare**:

| Metrică de Performanță | Scor Înregistrat | Descriere |
| :--- | :---: | :--- |
| **JSON Format Valid** | **98.6%** | Rata de răspunsuri generate direct ca obiecte JSON parsabile, fără text adiacent sau erori de sintaxă. |
| **Acuratețe Potrivire Rol** | **85.7%** | Procentul de identificare și clasificare corectă a titulaturii profesionale conform profilului real. |
| **Suprapunere Skill-uri (Jaccard Index)** | **68.5%** | Măsura strictă de suprapunere între tehnologiile extrase de model și cele din Ground Truth. |


