import pandas as pd
from flask import Flask, render_template, request, jsonify
from collections import Counter

app = Flask(__name__)

df = pd.read_csv("datos_farmacia_procesados.csv", low_memory=False)

codigo_a_nombre = dict(zip(df['codigo'], df['nombre']))
codigo_a_descripcion = dict(zip(df['codigo'], df['descripcion']))
codigo_a_precio = dict(zip(df['codigo'], df['ImporteCoste']))

unique_codes = set()
for codes in df['codigo']:
    unique_codes.update(codes)
unique_codes = sorted(unique_codes)


def get_most_frequent_products_by_name(reference_code_or_name):
    reference_code = None
    reference_name = None
    reference_full_description = None
    if str(reference_code_or_name).isdigit():
        reference_code = int(reference_code_or_name)
        reference_name = codigo_a_nombre.get(reference_code, None)
        reference_full_description = codigo_a_descripcion.get(reference_code, "No description available")
    else:
        for code, name in codigo_a_nombre.items():
            if reference_code_or_name.lower() in name.lower():
                reference_code = code
                reference_name = name
                reference_full_description = codigo_a_descripcion.get(code, "No description available")
                break
    if not reference_name:
        return {"reference_count": 0, "reference_full_description": "", "suggestions": []}
    reference_count = 0
    for codes_list in df['codigo']:
        for code in codes_list:
            if reference_name.lower() in str(codigo_a_nombre[code]).lower():
                reference_count += 1
    sales_with_reference = df[df['codigo'].apply(
        lambda codes: any(reference_name.lower() in str(codigo_a_nombre[code]).lower() for code in codes))]
    related_products = [
        product for codes_list in sales_with_reference['codigo']
        for product in codes_list if product != reference_code
    ]
    product_counts = Counter(related_products)
    candidates = product_counts.most_common(10)
    filtered = []
    for prod, _ in candidates:
        candidate_name = codigo_a_nombre[prod]
        if reference_name.lower() in candidate_name.lower():
            continue
        filtered.append(prod)
        if len(filtered) == 5:
            break
    suggestions = []
    for prod in filtered:
        suggestions.append({
            'codigo': prod,
            'nombre': str(codigo_a_nombre[prod]).capitalize(),
            'description': codigo_a_descripcion.get(prod, 'No description available'),
            'price': f"€{float(codigo_a_precio.get(prod, 0)):.2f}",
            'count': product_counts[prod]
        })
    return {"reference_count": reference_count,
            "reference_full_description": reference_full_description,
            "suggestions": suggestions}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    reference_code_or_name = data.get("product_code_or_name")
    search_option = data.get("search_option")

    if search_option == "Sin embeddings":
        result = get_most_frequent_products_by_name(reference_code_or_name)
    else:
        mapping = {
            "Embeddings Auto/TSNE": ("embeddings/embeddings_tsne_autoencoder.csv", "clusters/clusters_tsne_autoencoder.csv"),
            "Embeddings TSNE": ("embeddings/embeddings_tsne.csv", "clusters/clusters_tsne.csv"),
            "Embeddings NMF": ("embeddings/embeddings_nmf.csv", "clusters/clusters_nmf.csv"),
            "Embeddings PCA": ("embeddings/embeddings_pca.csv", "clusters/clusters_pca.csv")
        }
        if search_option in mapping:
            if not str(reference_code_or_name).isdigit():
                for code, name in codigo_a_nombre.items():
                    if reference_code_or_name.lower() in str(name).lower():
                        reference_code_or_name = code
                        break
            if not str(reference_code_or_name).isdigit() or int(reference_code_or_name) not in unique_codes:
                return jsonify({"error": "Código de referencia no encontrado en unique_codes."})
            reference_code = int(reference_code_or_name)
            embeddings_file, clusters_file = mapping[search_option]
            embeddings_df = pd.read_csv(embeddings_file, header=None)
            embeddings = embeddings_df.values
            clusters_df = pd.read_csv(clusters_file, header=None)
            clusters = clusters_df.values.flatten()
            from functs import find_top5_nearest_in_cluster
            results = find_top5_nearest_in_cluster(reference_code, embeddings, unique_codes, clusters, codigo_a_nombre)
            if results is None:
                return jsonify({"error": "No se encontraron productos cercanos en el mismo cluster."})

            suggestions = []
            for tup in results:
                prod_code = int(tup[0])
                suggestions.append({
                    'codigo': prod_code,
                    'nombre': str(codigo_a_nombre.get(prod_code, "Desconocido")).capitalize(),
                    'description': codigo_a_descripcion.get(prod_code, "No description available"),
                    'price': f"€{float(codigo_a_precio.get(prod_code, 0)):.2f}"
                })
            result = {"reference_count": None, "reference_full_description": None, "suggestions": suggestions}
        else:
            result = {"reference_count": 0, "reference_full_description": "", "suggestions": []}
    return jsonify(result)


if __name__ == "__main__":
    app.run()
