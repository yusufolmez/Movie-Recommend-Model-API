import pandas as pd
import requests


# Veri setini yükleme
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Veri seti başarıyla yüklendi: {len(df)} kayıt.")
        return df
    except FileNotFoundError:
        print("Veri seti dosyası bulunamadı. Yeni bir veri seti oluşturuluyor.")
        return pd.DataFrame(columns=[
            "id", "title", "vote_average", "vote_count", "status", "release_date", "revenue",
            "runtime", "adult", "backdrop_path", "budget", "homepage", "imdb_id",
            "original_language", "original_title", "overview", "popularity", "poster_path",
            "tagline", "genres", "production_companies", "production_countries",
            "spoken_languages", "keywords"
        ])


# Veri setini kaydetme
def save_dataset(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Veri seti başarıyla kaydedildi: {file_path}")


# Yeni filmleri ekleme
def add_movies_to_dataset(df, movies_data):
    added_count = 0
    for movie_data in movies_data:
        movie_id = movie_data['id']
        if movie_id not in df['id'].values:
            new_movie = {
                "id": movie_data['id'],
                "title": movie_data['title'],
                "vote_average": movie_data['vote_average'],
                "vote_count": movie_data['vote_count'],
                "status": movie_data['status'],
                "release_date": movie_data['release_date'],
                "revenue": movie_data['revenue'],
                "runtime": movie_data['runtime'],
                "adult": movie_data['adult'],
                "backdrop_path": movie_data['backdrop_path'],
                "budget": movie_data['budget'],
                "homepage": movie_data['homepage'],
                "imdb_id": movie_data['imdb_id'],
                "original_language": movie_data['original_language'],
                "original_title": movie_data['original_title'],
                "overview": movie_data['overview'],
                "popularity": movie_data['popularity'],
                "poster_path": movie_data['poster_path'],
                "tagline": movie_data.get('tagline', None),
                "genres": [genre['name'] for genre in movie_data.get('genres', [])],
                "production_companies": [company['name'] for company in movie_data.get('production_companies', [])],
                "production_countries": [country['name'] for country in movie_data.get('production_countries', [])],
                "spoken_languages": [language['name'] for language in movie_data.get('spoken_languages', [])],
                "keywords": movie_data.get('keywords', None),
            }
            new_movie_df = pd.DataFrame([new_movie])
            df = pd.concat([df, new_movie_df], ignore_index=True)
            added_count += 1
            print(f"Yeni film eklendi: {movie_data['title']} (ID: {movie_id})")
        else:
            print(f"Film zaten mevcut: {movie_data['title']} (ID: {movie_id})")
    print(f"Toplam eklenen film sayısı: {added_count}")
    return df, added_count


# TMDb API'den yeni filmleri çekme
def fetch_latest_movies(api_key, last_id):
    url = f"https://api.themoviedb.org/3/movie/latest?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        latest_movie = response.json()
        if latest_movie['id'] > last_id:
            print(f"Yeni film bulundu: {latest_movie['title']} (ID: {latest_movie['id']})")
            return [latest_movie]
        else:
            print("Yeni bir film bulunamadı.")
            return []
    else:
        print(f"API isteği başarısız oldu. Durum kodu: {response.status_code}")
        return []


# Ana fonksiyon
def main():
    # API anahtarı ve dosya yolu
    api_key = "1ac1c652640394393d245daab04c06b2"  # TMDB API anahtarınızı buraya ekleyin
    file_path = "TMDB_movie_dataset_v11.csv"

    # Veri setini yükle
    df = load_dataset(file_path)

    # Veri setindeki son ID'yi al
    last_id = df['id'].max() if not df.empty else 0
    print(f"Veri setindeki son ID: {last_id}")

    # Yeni filmleri çek
    new_movies = fetch_latest_movies(api_key, last_id)

    # Yeni filmleri ekle
    df, added_count = add_movies_to_dataset(df, new_movies)

    # Güncellenmiş veri setini kaydet
    save_dataset(df, file_path)

    print(f"Toplam {added_count} film veri setine eklendi.")


# Programı çalıştır
if __name__ == "__main__":
    main()
