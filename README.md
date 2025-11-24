# 🧠 Parkinson Voice Classifier  
Modelo de Machine Learning para classificação da Doença de Parkinson utilizando sinais de voz.

---

## 📌 Visão Geral
Este projeto implementa um classificador para identificar a presença de Parkinson a partir de **características acústicas extraídas de gravações de voz**.

Ele utiliza um dataset público amplamente usado em pesquisas sobre biomarcadores vocais para Parkinson, contendo atributos como jitter, shimmer, medidas de ruído e irregularidade vocal.

O código atual permite:
- carregar o dataset
- explorar características principais
- treinar um modelo de classificação
- avaliar o desempenho preditivo

---

## 📂 Estrutura do Repositório (Atual)

    parkinson-voice-classifier/
    │
    ├── extracted_features.csv # Arquivo com features pré-processadas
    ├── parkinsons_train.csv # Dataset original
    ├── main.py # Script principal com treino e avaliação
    └── README.md # Este arquivo


> 🔧 Obs.: A pasta `venv/` existe apenas localmente e **não deve ser versionada**.  
> Recomenda-se adicionar um `.gitignore` (posso gerar se quiser).

---

## 🚀 Tecnologias Utilizadas
- **Python 3.10+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn** (se usado)
- Ambiente virtual (`venv`)

---

## 🧬 Dataset
O projeto utiliza o arquivo:

- **`parkinsons_train.csv`**  
  Contém atributos vocais como:
  - Jitter (%)
  - Shimmer (dB)
  - NHR (Noise-to-Harmonics Ratio)
  - HNR
  - RPDE
  - DFA
  - Spread1 / Spread2
  - PPE  
  - `status` → variável alvo (0 = saudável / 1 = Parkinson)

Além disso, há o arquivo **`extracted_features.csv`**, que representa uma versão tratada ou reduzida do dataset.
