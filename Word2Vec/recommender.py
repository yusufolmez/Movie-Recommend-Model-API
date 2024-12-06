import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import pandas as pd

model = Word2Vec.load("word2vec_tmdb.model")
df = pd.read_csv("../filtered_movies.csv")

# Film özetlerini ve kullanıcı metnini vektöre dönüştürme fonksiyonları
def get_vector_from_summary(summary, model):
    words = summary.split()  # Özeti kelimelere ayır
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Kelime vektörlerinin ortalamasını al
    else:
        return np.zeros(model.vector_size)  # Eğer modelde olmayan kelimeler varsa, sıfır vektörü döndür


def get_vector_from_input(user_input, model):
    words = user_input.split()  # Girişi kelimelere ayır
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Kullanıcı girdisinin kelime vektörlerinin ortalamasını al
    else:
        return np.zeros(model.vector_size)  # Eğer modelde olmayan kelimeler varsa, sıfır vektörü döndür


# Benzerlik hesaplayarak en yakın filmleri önerme fonksiyonu
def recommend_movies_by_input(user_input, summaries, model, top_n=5):
    user_vector = get_vector_from_input(user_input, model)  # Kullanıcı girdisini vektöre dönüştür
    similarities = []

    for idx, summary in enumerate(summaries):
        summary_vector = get_vector_from_summary(summary, model)  # Her film özetini vektöre dönüştür
        similarity = cosine_similarity([user_vector], [summary_vector])[0][0]  # Benzerliği hesapla
        similarities.append((idx, similarity))  # İndeks ve benzerlik skorunu sakla

    # Benzerlik skorlarına göre sıralama
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # En benzer top_n filmi döndür
    return similarities[:top_n]


# En yüksek puanlı filmleri önerme fonksiyonu
def recommend_top_rated_movies(similar_movies, ratings, top_n=5):
    # En yüksek puana sahip filmleri filtrele
    top_movies = sorted(similar_movies, key=lambda x: ratings[x[0]], reverse=True)  # Puanlara göre sıralama
    return top_movies[:top_n]  # En yüksek 5 puanlı filmi döndür


film_summaries = df['overview'].dropna().tolist()
film_titles = df['title'].dropna().tolist()
film_ratings = df['vote_average'].dropna().tolist()  # Rating sütunu, puanların bulunduğu sütun olmalı

# Kullanıcıdan gelen metin girişi
user_input = "Action film set in the 1990s."

# Tavsiye sistemi kullanımı
recommended_movies = recommend_movies_by_input(user_input, film_summaries, model, top_n=20)

# En yüksek puanlı 5 filmi tavsiye et
top_rated_movies = recommend_top_rated_movies(recommended_movies, film_ratings, top_n=5)

print(user_input)
print("---------------------")
# Sonuçları göster
for idx, similarity in top_rated_movies:
    print(f"Recommended Movie {idx + 1}: {film_titles[idx]} - {film_summaries[idx]} with similarity score: {similarity} - Vote Average: {film_ratings[idx]}")
