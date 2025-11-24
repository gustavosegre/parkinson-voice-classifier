import os
import joblib
import numpy as np
from embeddings_wav2vec import extract_wav2vec_embedding

def predict_audio(model_path, encoder_path, wav_path, sr=16000, duration=5.0):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    emb = extract_wav2vec_embedding(wav_path, sr=sr, duration=duration).reshape(1, -1)
    pred = model.predict(emb)[0]
    try:
        proba = model.predict_proba(emb)[0][1]
    except Exception:
        proba = None

    label = encoder.inverse_transform([pred])[0]
    return {"file": os.path.basename(wav_path), "label": label, "proba_PD": proba}


if __name__ == "__main__":
    res = predict_audio(
        model_path="artifacts/svm_rbf_wav2vec.joblib",
        encoder_path="artifacts/label_encoder.joblib",
        wav_path="..wav" # arquivo .wav
    )
    print(res)
