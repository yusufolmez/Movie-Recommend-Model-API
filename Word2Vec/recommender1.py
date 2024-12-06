from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import pandas as pd

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Film Öneri API',
    description='Kullanıcı girdisine göre film önerileri sunan bir API')

ns = api.namespace('film-onerisi', description='Film öneri işlemleri')

# Model ve veri yükleme
try:
    model = Word2Vec.load("word2vec_tmdb.model")
    df = pd.read_csv("../tmdb_cleaned_no_revenue.csv")

    film_summaries = df['overview'].dropna().tolist()
    film_titles = df['title'].dropna().tolist()
    film_ratings = df['vote_average'].dropna().tolist()
    film_ids = df['id'].dropna().tolist()
except Exception as e:
    print(f"Model veya veri yüklenirken hata oluştu: {e}")

def get_vector_from_summary(summary, model):
    words = summary.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_vector_from_input(user_input, model):
    words = user_input.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def recommend_movies_by_input(user_input, summaries, model, top_n=5):
    user_vector = get_vector_from_input(user_input, model)
    similarities = []

    for idx, summary in enumerate(summaries):
        summary_vector = get_vector_from_summary(summary, model)
        similarity = cosine_similarity([user_vector], [summary_vector])[0][0]
        similarities.append((idx, similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def recommend_top_rated_movies(similar_movies, ratings, top_n=5):
    top_movies = sorted(similar_movies, key=lambda x: ratings[x[0]], reverse=True)
    return top_movies[:top_n]

# Input ve output modellerini tanımlama
input_model = api.model('Input', {
    'kullanici_girisi': fields.String(required=True, description='Kullanıcının film tercihleri'),
    'oneri_sayisi': fields.Integer(required=False, description='İstenilen öneri sayısı', default=5)
})

movie_model = api.model('Movie', {
    'id': fields.Integer(description='Film ID'),
    'baslik': fields.String(description='Film başlığı'),
    'ozet': fields.String(description='Film özeti'),
    'benzerlik_skoru': fields.Float(description='Benzerlik skoru'),
    'oy_ortalamasi': fields.Float(description='Ortalama oy')
})

output_model = api.model('Output', {
    'kullanici_girisi': fields.String(description='Kullanıcının girdiği metin'),
    'oneriler': fields.List(fields.Nested(movie_model), description='Önerilen filmler')
})

@ns.route('/')
class FilmOnerisi(Resource):
    @ns.doc('film_onerisi')
    @ns.expect(input_model)
    @ns.marshal_with(output_model, code=200, description='Başarılı')
    def post(self):
        """Film önerileri al"""
        data = request.json
        kullanici_girisi = data['kullanici_girisi']
        oneri_sayisi = data.get('oneri_sayisi', 5)

        try:
            # Kullanıcı girdisine göre benzer filmleri öner
            onerilen_filmler = recommend_movies_by_input(kullanici_girisi, film_summaries, model, top_n=20)
            en_iyi_filmler = recommend_top_rated_movies(onerilen_filmler, film_ratings, top_n=oneri_sayisi)

            # Sonuçları ekle
            sonuclar = []
            for idx, benzerlik in en_iyi_filmler:
                sonuclar.append({
                    "id": film_ids[idx],
                    "baslik": film_titles[idx],
                    "ozet": film_summaries[idx],
                    "benzerlik_skoru": benzerlik,
                    "oy_ortalamasi": film_ratings[idx]
                })

            return {
                "kullanici_girisi": kullanici_girisi,
                "oneriler": sonuclar
            }

        except Exception as e:
            api.abort(500, f"Bir hata oluştu: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
