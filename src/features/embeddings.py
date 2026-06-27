"""
Generate sentence-transformer embeddings from amenity text, reduce via PCA.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_COMPONENTS = 8


def amenities_to_text(amenity_str: str) -> str:
    import ast, re
    try:
        items = ast.literal_eval(amenity_str)
        return ", ".join(items)
    except Exception:
        return ""


def build_embeddings(
    df: pd.DataFrame,
    fit_pca: bool = True,
    pca: PCA | None = None,
    model: SentenceTransformer | None = None,
) -> tuple[np.ndarray, PCA, SentenceTransformer]:
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    texts = df["amenities"].apply(amenities_to_text).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    if fit_pca:
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = pca.transform(embeddings)

    return reduced, pca, model


def embeddings_to_df(reduced: np.ndarray) -> pd.DataFrame:
    cols = [f"embed_{i}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=cols)
