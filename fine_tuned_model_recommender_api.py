from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app)  # CORS desteği ekleniyor

# Veri setini yükleyelim
df = pd.read_csv('ttl_ovrwv.csv')  # Dosya yolunuza göre değiştirin
film_idleri = df['id'].tolist()
film_ozetleri = df['overview'].tolist()
film_isimleri = df['title'].tolist()

# Modeli yükleyin
model = SentenceTransformer('fine_tuned_model')

# Kaydedilen film embeddings dosyasını yükleyin
film_embeddings = torch.load('film_embeddings.pt', weights_only=True)

# API rotası
@app.route('/recommend', methods=['POST'])
def recommend():
    # Kullanıcı girdisini al
    data = request.json
    kullanici_input = data.get('input')

    if not kullanici_input:
        return jsonify({"error": "Input is required"}), 400

    # Giriş metnini encode et
    kullanici_embedding = model.encode([kullanici_input], convert_to_tensor=True)

    # Cosine similarity hesapla
    cosine_similarities = torch.nn.functional.cosine_similarity(kullanici_embedding, film_embeddings)

    # En benzer 5 filmi bulun
    en_benzer_indices = torch.topk(cosine_similarities, 5).indices.tolist()
    results = []
    for idx in en_benzer_indices:
        results.append({
            "id": film_idleri[idx],
            "film": film_isimleri[idx],
            "ozet": film_ozetleri[idx],
            "benzerlik_skoru": round(cosine_similarities[idx].item(), 2)
        })

    return jsonify({"results": results})

# Sunucuyu çalıştır
if __name__ == '__main__':
    app.run(debug=True, port=8080)
