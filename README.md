# Parkinson Voice Classifier  
Classificação de presença de Parkinson utilizando Machine Learning aplicado a sinais de voz.

---

## Sobre o Projeto
Este projeto tem como objetivo construir um modelo de Machine Learning capaz de identificar a presença de Parkinson baseado em **características extraídas de sinais de voz**, seguindo abordagens modernas de pré-processamento, engenharia de atributos e modelos supervisionados.

O projeto é inspirado em pesquisas recentes que associam padrões vocais às manifestações motoras da doença.

---

## Tecnologias Utilizadas
- **Python 3.10+**
- **Pandas / NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**
- (Opcional futuro) **TensorFlow / PyTorch** para redes neurais

---

## Estrutura do Repositório
    
    parkinson_voice_classifier/
    │
    ├── data/
    │ ├── raw/ # Arquivos originais (não alterados)
    │ ├── processed/ # Dados tratados
    │
    ├── notebooks/
    │ ├── 01_exploration.ipynb
    │ ├── 02_training.ipynb
    │ └── 03_evaluation.ipynb
    │
    ├── src/
    │ ├── preprocessing.py
    │ ├── features.py
    │ ├── model.py
    │ └── utils.py
    │
    ├── venv/ # Ambiente virtual (não sobe para o GitHub)
    ├── requirements.txt
    └── README.md


---

## 🧬 Dataset
O projeto utiliza um dataset de sinais de voz anotados com a presença (`1`) ou ausência (`0`) de Parkinson.

O dataset inclui atributos como:
- jitter  
- shimmer  
- HNR  
- RPDE  
- DFA  
- entre outros parâmetros acústicos do sinal.

Obs: Este repositório **não contém os dados brutos** por questões de licença.  
O usuário deve colocar o arquivo na pasta `data/raw/`.

---

## Pré-processamento
As principais etapas empregadas:

- Remoção de valores ausentes  
- Normalização (StandardScaler ou MinMaxScaler)  
- Seleção de atributos relevantes  
- Divisão em treino/teste  
- Redução de dimensionalidade (opcional)  

---

## Modelos Utilizados
Atualmente testados:

- **Random Forest**
- **SVM**
- **Logistic Regression**
- **MLP Classifier**
- **KNN**

Métricas de avaliação:

- Acurácia  
- F1-Score  
- Matriz de confusão  
- ROC AUC  

