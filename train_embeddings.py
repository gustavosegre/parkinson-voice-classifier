import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

from embeddings_wav2vec import build_embeddings

SEED = 42
CV_K = 5


def get_models():
    models = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=3.0, kernel="rbf", gamma="scale",
                probability=True, class_weight="balanced", random_state=SEED
            ))
        ])
    }
    if XGB_OK:
        models["xgb"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED
        )
    return models


def main():
    df = build_embeddings(base_path="data/raw", sr=16000, duration=5.0)
    print("Shape:", df.shape)

    meta_cols = [c for c in ["class", "task", "file", "patient", "group"] if c in df.columns]
    X = df.drop(columns=meta_cols).values

    y = df["class"].values
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )

    models = get_models()
    os.makedirs("artifacts", exist_ok=True)

    for name, model in models.items():
        print(f"\n=== Treinando {name} ===")
        skf = StratifiedKFold(n_splits=CV_K, shuffle=True, random_state=SEED)
        cv_results = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False
        )
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"{k:>9}: {cv_results['test_'+k].mean():.4f} ± {cv_results['test_'+k].std():.4f}")

        # Treino final
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            print("ROC-AUC (teste):", roc_auc_score(y_test, y_proba))
        except Exception:
            pass

        print(classification_report(y_test, y_pred, target_names=encoder.classes_))
        print(confusion_matrix(y_test, y_pred))

        joblib.dump(model, f"artifacts/{name}_wav2vec.joblib")
        joblib.dump(encoder, "artifacts/label_encoder.joblib")

    df.to_csv("artifacts/voice_embeddings_dataset.csv", index=False)


if __name__ == "__main__":
    main()
