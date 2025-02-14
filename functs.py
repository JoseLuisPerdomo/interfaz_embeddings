import time
from collections import Counter
from statistics import mean, mode

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
import plotly.graph_objects as go
from sklearn.decomposition import NMF


def embeddings_cluster_graph_pca(n_components, num_clusters, co_matrix, unique_codes, codigo_a_nombre):
    pca = PCA(n_components=n_components)
    product_embeddings = pca.fit_transform(co_matrix)
    embeddings_3d = normalize(product_embeddings, axis=1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_3d)  # Asegúrate de que embeddings_3d tenga 3 dimensiones

    # Colores para los clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown']

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
    return clusters, embeddings_3d


def embeddings_cluster_graph_nmf(n_components, num_clusters, co_matrix, unique_codes, codigo_a_nombre):
    nmf = NMF(n_components=n_components, random_state=42)
    product_embeddings = nmf.fit_transform(co_matrix)
    embeddings_3d = normalize(product_embeddings, axis=1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_3d)  # Asegúrate de que embeddings_3d tenga 3 dimensiones

    # Colores para los clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown']

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
    return clusters, embeddings_3d


def embeddings_cluster_graph_lsa(graph_dim, n_components, num_clusters, co_matrix, unique_codes, codigo_a_nombre):
    lsa = FastICA(n_components=n_components, random_state=42)
    product_embeddings = lsa.fit_transform(co_matrix)
    product_embeddings = normalize(product_embeddings, axis=1)

    tsne = TSNE(n_components=graph_dim, random_state=42, init='random')
    embeddings_3d = tsne.fit_transform(product_embeddings)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_3d)  # Asegúrate de que embeddings_3d tenga 3 dimensiones

    # Colores para los clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown']

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
    fig.show()


def df_32_cols():
    df = pd.read_csv("datos_farmacia.csv", low_memory=False)
    df = df[pd.to_numeric(df['codigo'], errors='coerce').notna()]
    df['codigo'] = df['codigo'].astype(int)
    df = df.loc[:, ~(df.isna() | (df == 0)).all()]
    df = df.drop(columns=['Fecha', 'PvpOld', 'Xclie_IdCliente', 'Cliente', 'TipoCliente', 'entregado', 'TIPOTARIFA',
                          'ColorArticu', 'ColorCategoria', 'ColorCliente', 'ColorFamilia', 'ColorLista',
                          'linea_idventa'])

    df[['DescuentoLinea', 'DescuentoOpera', 'Entrega', 'Devuelto']] = df[
        ['DescuentoLinea', 'DescuentoOpera', 'Entrega', 'Devuelto']].fillna(0.0)
    df['aportacion'] = df['aportacion'].fillna('A')
    df = df.dropna(subset=['RecetaPendiente', 'NombreFamilia'])

    return df


def generar_embeddings_compuesto(df, columnas_numericas, columnas_texto, columnas_categoricas):
    n = len(df)

    if columnas_numericas:
        scaler = StandardScaler()
        numeric_vals = scaler.fit_transform(df[columnas_numericas])
        emb_numerico = csr_matrix(numeric_vals)
    else:
        emb_numerico = csr_matrix((n, 0))

    if columnas_categoricas:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        emb_categorico = encoder.fit_transform(df[columnas_categoricas])
    else:
        emb_categorico = csr_matrix((n, 0))

    if columnas_texto:
        modelo_texto = SentenceTransformer('all-MiniLM-L6-v2')
        text_list = df[columnas_texto[0]].astype(str).tolist()
        emb_texto = modelo_texto.encode(text_list)
        emb_texto = csr_matrix(emb_texto)
    else:
        emb_texto = csr_matrix((n, 0))

    embeddings_finales = hstack([emb_numerico, emb_categorico, emb_texto])

    return embeddings_finales


def df_agrupado_ventas():
    df = pd.read_csv("datos_farmacia.csv", low_memory=False)

    df = df[pd.to_numeric(df['codigo'], errors='coerce').notna()]
    df['codigo'] = df['codigo'].astype(int)
    df = df.loc[:, ~(df.isna() | (df == 0)).all()]
    df = df.drop(columns=['Fecha', 'PvpOld', 'Xclie_IdCliente', 'Cliente', 'TipoCliente', 'entregado', 'TIPOTARIFA',
                          'ColorArticu', 'ColorCategoria', 'ColorCliente', 'ColorFamilia', 'ColorLista',
                          'linea_idventa', 'familiavend', 'idvendedor', 'TipoLineaVenta', 'TipoPago', 'NumCaja',
                          'DescuentoLinea', 'DescuentoOpera', 'Entrega', 'Devuelto', 'Importebruto', 'ImporteCoste',
                          'NumeroDoc', 'esfactura', 'TotalVenta', 'RecetaPendiente', 'NombreFamilia', 'Precio',
                          'Familia', 'tipolinea', 'Facturada', 'aportacion', 'TipoVenta', 'TotalVentaBruta',
                          'ImportePmc', 'ImporteNeto', 'idnlinea', 'unidades', 'dtolineaventa', 'dtooperaventa'])

    df['nombre'] = df['descripcion'].apply(lambda x: x.split()[0] if isinstance(x, str) else '')
    codigo_a_nombre = dict(zip(df['codigo'], df['nombre']))
    codigo_a_descripcion = dict(zip(df['codigo'], df['descripcion']))

    df = df.groupby('IdVenta').agg(list).reset_index()

    return df, codigo_a_nombre, codigo_a_descripcion


def get_most_frequent_products(df, reference_code):
    sales_with_reference = df[df['codigo'].apply(lambda codes: reference_code in codes)]
    related_products = [
        product for codes_list in sales_with_reference['codigo']
        for product in codes_list if product != reference_code
    ]
    product_counts = Counter(related_products)
    return product_counts.most_common(10)


def benchmark_product_suggestions(df, product_codes):
    execution_times = []
    product_suggestions = {}

    for code in product_codes:
        start_time = time.perf_counter()
        product_suggestions[code] = get_most_frequent_products(df, code)
        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) * 1000
        execution_times.append((code, elapsed_time))

    df_execution_times = pd.DataFrame(execution_times, columns=['Product_Code', 'Execution_Time_ms'])

    avg_time = mean(df_execution_times['Execution_Time_ms'])
    max_time = max(df_execution_times['Execution_Time_ms'])
    min_time = min(df_execution_times['Execution_Time_ms'])
    mode_time = mode(df_execution_times['Execution_Time_ms'])

    print("\n--- Execution Time Report ---")
    print(f"Average Time: {avg_time:.2f} ms")
    print(f"Maximum Time: {max_time:.2f} ms")
    print(f"Minimum Time: {min_time:.2f} ms")
    print(f"Mode Time: {mode_time:.2f} ms")

    return df_execution_times


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