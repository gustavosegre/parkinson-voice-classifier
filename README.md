from pathlib import Path

readme_content = """# 🧠 Detecção de Parkinson via Voz com Wav2Vec2 + SVM

Este projeto implementa um pipeline completo de **classificação de voz** para **detecção de Parkinson**, utilizando embeddings extraídos do modelo **Wav2Vec2** da Meta (Facebook AI).  
O sistema processa arquivos de áudio `.wav`, gera embeddings de alto nível e treina classificadores supervisionados (SVM e opcionalmente XGBoost) para distinguir **indivíduos saudáveis (HC)** de **pacientes com Parkinson (PD)**.

---

## 📁 Estrutura do Projeto

