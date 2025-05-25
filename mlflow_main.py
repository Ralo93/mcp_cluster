import os
import json
import numpy as np
import mlflow
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import hmean
from umap import UMAP
from hdbscan import HDBSCAN
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from data_loader import DataLoader, Speech, build_db_url

DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'database': 'cluster',
    'document_table': 'speeches'
}
DB_URL = build_db_url(DB_CONFIG)
CSV_PATH = "data/leg20.csv"

import plotly.express as px
import pandas as pd

def visualize_clusters(embeddings, labels, texts, original_umap_params, max_text_len=150):
    """
    Runs UMAP to 2D if necessary and visualizes clustered embeddings using Plotly.
    """
    from umap import UMAP
    import plotly.express as px
    import pandas as pd

    # Re-run UMAP if n_components != 2
    if original_umap_params['n_components'] != 2:
        reducer_vis = UMAP(
            n_components=2,
            min_dist=original_umap_params['min_dist'],
            metric=original_umap_params['umap_metric'],
            random_state=42
        )
        embeddings_2d = reducer_vis.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df["cluster"] = labels
    df["text"] = [t[:max_text_len] + "..." if len(t) > max_text_len else t for t in texts]

    df = df[df["cluster"] != -1]  # Exclude noise

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["cluster"].astype(str),
        hover_data=["text"],
        title="Interactive Cluster Visualization (UMAP 2D Projection)",
        labels={"color": "Cluster"}
    )
    fig.update_layout(legend_title_text='Cluster', width=1000, height=700)
    fig.show()



def load_embeddings():
    loader = DataLoader(CSV_PATH, DB_CONFIG)
    speeches_df = loader.load_from_db()
    speech_objs = []

    for _, row in speeches_df.iterrows():
        speech = Speech(
            content=row['topic_desc'],
            politician=row['politician'],
            term=row['term'],
            position=row['position'],
            date=row['date'],
            faction=row['faction'],
            embedding=row['embedding']
        )
        speech_objs.append(speech)

    embeddings = np.array([s.embedding for s in speech_objs])
    return embeddings, [s.content for s in speech_objs]


def evaluate_clustering(embeddings_2d, labels):
    mask = labels != -1
    if len(set(labels[mask])) <= 1:
        return -1, -1, -1, -1

    sil = silhouette_score(embeddings_2d[mask], labels[mask])
    ch = calinski_harabasz_score(embeddings_2d[mask], labels[mask])
    db = davies_bouldin_score(embeddings_2d[mask], labels[mask])
    harmonic = hmean([sil, ch, 1.0 / (db + 1e-6)])
    return sil, ch, db, harmonic


def run_clustering_experiment(embeddings, texts, min_cluster_size, min_dist, n_components, umap_metric="cosine"):
    mlflow.set_experiment("german_speeches_clustering_test_visualization2")

    with mlflow.start_run():
        mlflow.log_param("min_cluster_size", min_cluster_size)
        mlflow.log_param("umap_min_dist", min_dist)
        mlflow.log_param("umap_n_components", n_components)
        mlflow.log_param("umap_metric", umap_metric)

        reducer = UMAP(n_components=n_components, min_dist=min_dist, metric=umap_metric, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(embeddings_2d)

        sil, ch, db, harmonic = evaluate_clustering(embeddings_2d, labels)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("calinski_harabasz", ch)
        mlflow.log_metric("davies_bouldin", db)
        mlflow.log_metric("harmonic_mean", harmonic)
        mlflow.log_metric("num_clusters", num_clusters)
        mlflow.log_metric("noise_ratio", noise_ratio)

        print("\n================= Experiment Results =================")
        print(f"min_cluster_size      : {min_cluster_size}")
        print(f"UMAP min_dist         : {min_dist}")
        print(f"UMAP n_components     : {n_components}")
        print(f"UMAP metric           : {umap_metric}")
        print(f"Silhouette Score      : {sil:.4f}")
        print(f"Calinski-Harabasz     : {ch:.4f}")
        print(f"Davies-Bouldin        : {db:.4f}")
        print(f"Harmonic Mean         : {harmonic:.4f}")
        print(f"Number of Clusters    : {num_clusters}")
        print(f"Noise Ratio           : {noise_ratio:.2%}")
        print("====================================================\n")

    return -harmonic  # for minimization


if __name__ == "__main__":
    embeddings, texts = load_embeddings()

    space = [
        Integer(30, 200, name='min_cluster_size'),
        Real(0.0, 0.5, name='min_dist'),
        Integer(2, 15, name='n_components'),
        Categorical(['cosine', 'euclidean'], name='umap_metric')
    ]

    @use_named_args(space)
    def objective(**params):
        return run_clustering_experiment(
            embeddings,
            texts,
            min_cluster_size=params['min_cluster_size'],
            min_dist=params['min_dist'],
            n_components=params['n_components'],
            umap_metric=params['umap_metric']
        )

    result = gp_minimize(objective, space, n_calls=10, random_state=42)

    import joblib

    # After optimization is done
    best_params = {
        'min_cluster_size': result.x[0],
        'min_dist': result.x[1],
        'n_components': result.x[2],
        'umap_metric': result.x[3],
    }

    # Train final model using best parameters
    mlflow.set_experiment("german_speeches_clustering_bayesian")
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)

        reducer = UMAP(
            n_components=best_params['n_components'],
            min_dist=best_params['min_dist'],
            metric=best_params['umap_metric'],
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(embeddings)

        clusterer = HDBSCAN(min_cluster_size=best_params['min_cluster_size'], metric="euclidean")
        labels = clusterer.fit_predict(embeddings_2d)

        sil, ch, db, harmonic = evaluate_clustering(embeddings_2d, labels)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        mlflow.log_metrics({
            "silhouette_score": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "harmonic_mean": harmonic,
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
        })

        # Save models
        os.makedirs("models", exist_ok=True)
        joblib.dump(reducer, "models/umap_model.joblib")
        joblib.dump(clusterer, "models/hdbscan_model.joblib")

        # Log artifacts
        mlflow.log_artifact("models/umap_model.joblib", artifact_path="models")
        mlflow.log_artifact("models/hdbscan_model.joblib", artifact_path="models")

        print("✅ Saved best model artifacts to MLflow.")

        visualize_clusters(embeddings, labels, texts, best_params)



