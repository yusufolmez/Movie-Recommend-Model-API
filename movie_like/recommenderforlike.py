import pickle

import gensim
import pymysql
import numpy as np
import pandas as pd

# Özet vektörlerini veritabanına kaydetme fonksiyonu
def save_summary_vectors_to_db(df, model, db_connection):
    cursor = db_connection.cursor()

    for _, row in df.iterrows():
        title = row['title']
        overview = row['overview']

        # Film özetinden vektör hesapla
        vector = get_vector_from_summary(overview, model)

        # Vektörü BLOB formatında serileştir (pickle ile)
        vector_serialized = pickle.dumps(vector)  # Serialize et

        # SQL sorgusu ile veritabanına kaydet
        query = "INSERT INTO film_vectors (title, overview_vector) VALUES (%s, %s)"
        cursor.execute(query, (title, vector_serialized))

    db_connection.commit()
    cursor.close()

def get_vector_from_summary(summary, model):
    if not isinstance(summary, str):  # Eğer özet metin değilse
        return np.zeros(model.vector_size)  # Boş bir vektör döndür
    words = summary.split()  # Özeti kelimelere ayır
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Kelime vektörlerinin ortalamasını al
    else:
        return np.zeros(model.vector_size)  # Eğer modelde olmayan kelimeler varsa, sıfır vektörü döndür


# MySQL veritabanı bağlantısı için fonksiyon
def create_db_connection():
    return pymysql.connect(
        host='localhost',       # Veritabanı sunucu adı
        user='root',            # Veritabanı kullanıcı adı
        password='Yusuf.6707', # Veritabanı şifresi
        database='movie_like'   # Veritabanı adı
    )

# Film vektörleri tablosunun yapısını oluşturma
def create_table():
    db_connection = create_db_connection()
    cursor = db_connection.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS film_vectors (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        overview_vector BLOB
    );
    ''')
    db_connection.commit()
    cursor.close()

def main():
    # DataFrame'inizi yükleyin (örneğin CSV dosyasından)
    df = pd.read_csv('../TMDB_movie_dataset_v11.csv')  # CSV dosyasının yolunu değiştirin

    # Eğitilmiş Word2Vec modelini .model uzantılı dosya ile yükleyin
    model = gensim.models.Word2Vec.load('../word2vec_tmdb.model')  # .model uzantılı dosyayı yükleyin

    # Veritabanı bağlantısını oluşturun
    db_connection = create_db_connection()

    # Film vektörleri tablosunu oluşturun (ilk kez çalıştırıldığında)
    create_table()

    # Özet vektörlerini veritabanına kaydedin
    save_summary_vectors_to_db(df, model, db_connection)

    db_connection.close()

# Ana fonksiyonu çalıştır
if __name__ == '__main__':
    main()