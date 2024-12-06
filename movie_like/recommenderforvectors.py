from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS  # CORS modülünü ekleyin
import pymysql
import requests
from datetime import datetime

# TMDb API Key
API_KEY = '1ac1c652640394393d245daab04c06b2'

# Veritabanı Bağlantısı
def create_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='Yusuf.6707',
        database='movie_like'
    )


def get_movie_details_from_tmdb(movie_id):
    """
    TMDb API'den bir filmin detaylarını alır (IMDB ID dahil).
    """
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=external_ids'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for movie ID {movie_id}: {response.status_code}")
        return None


# Kullanıcının Beğendiği Filmleri Çekme
def get_user_liked_movies(user_sub):
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            query = """
            SELECT movie_id
            FROM user_liked_movies
            WHERE user_sub = %s AND is_liked = 1
            """
            cursor.execute(query, (user_sub,))
            result = cursor.fetchall()
            liked_movie_ids = [row[0] for row in result]
            return liked_movie_ids
    finally:
        connection.close()

# TMDb API'den Benzer Filmleri Çekme
def get_similar_movies_from_tmdb(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print(f"Error fetching similar movies for ID {movie_id}: {response.status_code}")
        return []

# Kullanıcının Beğenilerine Göre Benzer Filmler Önerme
def recommend_movies(user_sub):
    """
    Kullanıcının beğenilerine göre film önerileri yapar ve IMDB ID'sini ekler.
    """
    liked_movie_ids = get_user_liked_movies(user_sub)

    if not liked_movie_ids:
        return {"message": "No liked movies found for this user."}

    recommended_movies = []

    for movie_id in liked_movie_ids:
        similar_movies = get_similar_movies_from_tmdb(movie_id)
        recommended_movies.extend(similar_movies)

    # Filmleri filtrele: Puanı 5.5'in altındaki filmleri çıkar
    filtered_movies = [
        movie for movie in recommended_movies
        if movie.get('vote_average', 0) >= 5.5
    ]

    # Filmleri benzersiz hale getir (ID'ye göre)
    unique_movies = {movie['id']: movie for movie in filtered_movies}.values()

    # IMDB ID'sini eklemek için detayları al
    enriched_movies = []
    for movie in unique_movies:
        details = get_movie_details_from_tmdb(movie['id'])
        if details:
            movie['imdb_id'] = details.get('external_ids', {}).get('imdb_id')
        enriched_movies.append(movie)

    # Filmleri sıralama: Önce puan, sonra çıkış tarihi
    sorted_movies = sorted(
        enriched_movies,
        key=lambda x: (
            -x.get('vote_average', 0),  # Yüksek puan öncelikli
            datetime.strptime(x.get('release_date', '1900-01-01'), '%Y-%m-%d')  # Yeni tarih öncelikli
        )
    )

    return sorted_movies[:60]  # En iyi 10 filmi döndür

# Flask ve Flask-RESTx Ayarları
app = Flask(__name__)
CORS(app)  # CORS modülünü burada etkinleştirin
api = Api(app, version='1.0', title='Movie Recommendation API',
          description='API for recommending movies based on user preferences.')

ns = api.namespace('recommendations', description='Movie Recommendations')

# Swagger Model Tanımlamaları
recommendation_model = api.model('Recommendation', {
    'user_sub': fields.String(required=True, description='User identifier (Auth0 sub)'),
})

movie_model = api.model('Movie', {
    'id': fields.Integer(description='TMDb movie ID'),
    'title': fields.String(description='Movie title'),
    'overview': fields.String(description='Movie overview'),
    'popularity': fields.Float(description='Movie popularity'),
    'vote_average': fields.Float(description='Average vote'),
    'vote_count': fields.Integer(description='Number of votes'),
    'release_date': fields.String(description='Release date'),
})

# API Endpoints
@ns.route('/')
class RecommendationResource(Resource):
    @ns.expect(recommendation_model, validate=True)
    @ns.marshal_list_with(movie_model)
    def post(self):
        """
        Get movie recommendations based on user preferences.
        """
        data = api.payload
        user_sub = data.get('user_sub')
        recommendations = recommend_movies(user_sub)
        return recommendations, 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)
