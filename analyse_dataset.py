import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Setăm stilul graficelor
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

def analyze_cleaned_jsonl(file_path: str):
    print(f"📊 Se analizează fișierul: {file_path}...\n")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                text = item.get("Text", "")
                category = item.get("Category", "Unknown")
                
                char_count = len(text)
                word_count = len(text.split())
                
                data.append({
                    "Category": category,
                    "Char_Count": char_count,
                    "Word_Count": word_count
                })
            except json.JSONDecodeError:
                pass

    df = pd.DataFrame(data)
    
    # 1. Afișăm Statistici Descriptive
    print("="*50)
    print("📈 STATISTICI DESCRIPTIVE (LUNGIME TEXT & CUVINTE)")
    print("="*50)
    print(df[["Char_Count", "Word_Count"]].describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"}))
    print("\n")

    # Creăm o figură cu 4 sub-grafice
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Analiza Exploratorie a Datelor (EDA) - Resume Dataset", fontsize=16, fontweight="bold")

    # Grafic 1: Histograma pentru lungimea textului (Caractere)
    sns.histplot(df["Char_Count"], bins=50, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Distribuția Lungimii Textului (Caractere)")
    axes[0, 0].set_xlabel("Număr de Caractere")
    axes[0, 0].set_ylabel("Frecvență")
    
    # Adăugăm o linie pentru mediana
    median_chars = df["Char_Count"].median()
    axes[0, 0].axvline(median_chars, color="red", linestyle="--", label=f"Mediană: {int(median_chars)} car.")
    axes[0, 0].legend()

    # Grafic 2: Histograma pentru numărul de cuvinte
    sns.histplot(df["Word_Count"], bins=50, kde=True, ax=axes[0, 1], color="teal")
    axes[0, 1].set_title("Distribuția Numărului de Cuvinte per CV")
    axes[0, 1].set_xlabel("Număr de Cuvinte")
    axes[0, 1].set_ylabel("Frecvență")
    
    median_words = df["Word_Count"].median()
    axes[0, 1].axvline(median_words, color="red", linestyle="--", label=f"Mediană: {int(median_words)} cuvinte")
    axes[0, 1].legend()

    # Grafic 3: Top 15 Categorii
    top_categories = df["Category"].value_counts().head(15)
    sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Top 15 Categorii în Dataset")
    axes[1, 0].set_xlabel("Număr de CV-uri")
    axes[1, 0].set_ylabel("Categorie")

    # Grafic 4: Boxplot Lungime Text pe Top 10 Categorii (detectare outliers)
    top_10_cat_names = df["Category"].value_counts().head(10).index
    df_top10 = df[df["Category"].isin(top_10_cat_names)]
    
    sns.boxplot(data=df_top10, x="Char_Count", y="Category", ax=axes[1, 1], palette="mako")
    axes[1, 1].set_title("Distribuția Lungimii Textului pe Top 10 Categorii")
    axes[1, 1].set_xlabel("Număr de Caractere")
    axes[1, 1].set_ylabel("")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Salvare grafice într-o imagine HD
    output_img = "dataset_eda_report.png"
    plt.savefig(output_img, dpi=300)
    print(f"🖼️ Raportul grafic a fost salvat ca imagine în: {output_img}")
    plt.show()

if __name__ == "__main__":
    analyze_cleaned_jsonl("resumes_step1_cleaned2.jsonl")