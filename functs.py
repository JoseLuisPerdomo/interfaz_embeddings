import time
from collections import Counter
from statistics import mean, mode

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
from sklearn.decomposition import NMF


def embeddings_cluster_graph_pca(n_components, num_clusters, co_matrix, unique_codes, codigo_a_nombre):
    pca = PCA(n_components=n_components)
    product_embeddings = pca.fit_transform(co_matrix)
    embeddings_3d = normalize(product_embeddings, axis=1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_3d)  # Asegúrate de que embeddings_3d tenga 3 dimensiones

    # Colores para los clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown']
    """
    # Crear figura interactiva con Plotly
    fig = go.Figure()

    for i in range(num_clusters):
        indices = np.where(clusters == i)[0]

        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[indices, 0],
            y=embeddings_3d[indices, 1],
            z=embeddings_3d[indices, 2],
            mode='markers',
            marker=dict(size=6, color=colors[i], opacity=1.0),
            name=f'Cluster {i}',
            text=[f"Código: {unique_codes[idx]}<br>Nombre: {codigo_a_nombre.get(unique_codes[idx], 'Desconocido')}" for
                  idx in indices],
            hoverinfo="text"
        ))

    # Personalizar diseño
    fig.update_layout(
        title="Clusters de productos en 3D usando K-means y TSNE",
        scene=dict(
            xaxis_title="Componente 1",
            yaxis_title="Componente 2",
            zaxis_title="Componente 3"
        ),
        template="plotly_dark"
    )

    # Mostrar gráfico interactivo
    #fig.show()
    """
    return clusters, embeddings_3d


def embeddings_cluster_graph_nmf(n_components, num_clusters, co_matrix, unique_codes, codigo_a_nombre):
    nmf = NMF(n_components=n_components, random_state=42)
    product_embeddings = nmf.fit_transform(co_matrix)
    embeddings_3d = normalize(product_embeddings, axis=1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_3d)  # Asegúrate de que embeddings_3d tenga 3 dimensiones

    # Colores para los clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown']
    """
    # Crear figura interactiva con Plotly
    fig = go.Figure()

    for i in range(num_clusters):
        indices = np.where(clusters == i)[0]

        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[indices, 0],
            y=embeddings_3d[indices, 1],
            z=embeddings_3d[indices, 2],
            mode='markers',
            marker=dict(size=6, color=colors[i], opacity=1.0),
            name=f'Cluster {i}',
            text=[f"Código: {unique_codes[idx]}<br>Nombre: {codigo_a_nombre.get(unique_codes[idx], 'Desconocido')}" for
                  idx in indices],
            hoverinfo="text"
        ))

    # Personalizar diseño
    fig.update_layout(
        title="Clusters de productos en 3D usando K-means y TSNE",
        scene=dict(
            xaxis_title="Componente 1",
            yaxis_title="Componente 2",
            zaxis_title="Componente 3"
        ),
        template="plotly_dark"
    )

    # Mostrar gráfico interactivo
    #fig.show()
    """
    return clusters, embeddings_3d


def get_most_frequent_products(df, reference_code):
    sales_with_reference = df[df['codigo'].apply(lambda codes: reference_code in codes)]
    related_products = [
        product for codes_list in sales_with_reference['codigo']
        for product in codes_list if product != reference_code
    ]
    product_counts = Counter(related_products)
    return product_counts.most_common(10)


def get_most_frequent_products_by_name(df, reference_name):
    sales_with_reference = df[df['nombre'].apply(lambda x: reference_name.upper() in x.upper())]

    related_products = [
        product for names_list in sales_with_reference['nombre']
        for product in names_list if product != reference_name
    ]

    product_counts = Counter(related_products)

    return [prod for prod, _ in product_counts.most_common(5)]


def find_top5_nearest_in_cluster(reference_code, all_embeddings, unique_codes, clusters, codigo_a_nombre):
    try:
        ref_idx = unique_codes.index(reference_code)
    except ValueError:
        print("Código de referencia no encontrado en unique_codes.")
        return None

    ref_cluster = clusters[ref_idx]
    cluster_indices = np.where(clusters == ref_cluster)[0]
    cluster_indices = cluster_indices[cluster_indices != ref_idx]

    if len(cluster_indices) == 0:
        print("No hay otros productos en el mismo cluster.")
        return None

    ref_embedding = all_embeddings[ref_idx]
    distances = np.linalg.norm(all_embeddings[cluster_indices] - ref_embedding, axis=1)

    sorted_order = np.argsort(distances)
    top5_indices = cluster_indices[sorted_order[:5]]

    results = []
    for idx in top5_indices:
        code = unique_codes[idx]
        name = codigo_a_nombre.get(code, "Desconocido")
        results.append((code, name))

    return results
