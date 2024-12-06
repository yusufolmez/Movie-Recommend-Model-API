import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# nltk için gerekli veri setlerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# TMDb veri setini yükleyelim
df = pd.read_csv('../TMDB_movie_dataset_v11.csv')

# overview ve title sütunlarını alalım ve boş değerleri çıkaralım
df = df[['id', 'overview']].dropna()


# Ön işleme fonksiyonu: kelimeleri temizleyelim
def preprocess_text(text):
    # Küçük harflere çevir
    text = text.lower()
    # Noktalama işaretlerini çıkar
    text = re.sub(r'[^\w\s]', '', text)
    # Rakamları çıkar
    text = re.sub(r'\d+', '', text)
    # Tokenize et
    words = nltk.word_tokenize(text)
    # Stopwords'leri çıkar
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization (kelimeleri köklerine indir)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


# Özet vektörlerini kaydetmek için dosya adı
vector_file = 'overview_vectors.csv'
npy_file = 'overview_vectors.npy'

# Eğer özet vektörleri dosyası yoksa modeli eğit
if not os.path.exists('word2vec_tmdb.model'):
    print("Model bulunmuyor. Eğitiliyor...")

    # Tüm overview'leri işleyelim
    df['processed_overview'] = df['overview'].apply(lambda x: preprocess_text(x))

    # Word2Vec modelini eğitmek için liste formatına çeviriyoruz
    overviews_list = df['processed_overview'].tolist()

    # Word2Vec modelini eğitelim
    model = Word2Vec(
        sentences=overviews_list,  # Özetler
        vector_size=200,  # Vektör boyutunu artırıyoruz (daha fazla boyut, daha ayrıntılı ilişkiler)
        window=5,  # Daha dar bir bağlam aralığı
        min_count=5,  # Nadiren görülen kelimeleri çıkarıyoruz
        workers=4  # Paralel işlem gücü kullanımı
    )

    # Modeli kaydet
    model.save("word2vec_tmdb.model")
else:
    print("Model bulunuyor. Yeniden eğitilmeyecek.")
    model = Word2Vec.load("word2vec_tmdb.model")


# Vektörleri kaydetmek için fonksiyon
def save_overview_vectors(df):
    # İşlenmiş özet vektörlerini alalım
    df['overview_vectors'] = get_overview_vectors(df['processed_overview'])

    # Eğer dosya varsa, mevcut dosyayı yükleyip güncelleme yapalım
    if os.path.exists(vector_file):
        existing_df = pd.read_csv(vector_file)
        # Eski ve yeni veriyi birleştirelim
        combined_df = pd.concat([existing_df, df[['id', 'overview_vectors']]], ignore_index=True)
        # Sadece benzersiz başlıkları tutacak şekilde sıralayıp güncelleme yapalım
        combined_df = combined_df.drop_duplicates(subset='id').reset_index(drop=True)
        combined_df.to_csv(vector_file, index=False)
    else:
        # Dosya yoksa, ilk kez kaydedelim
        df[['id', 'overview_vectors']].to_csv(vector_file, index=False)

    # NumPy dosyasına kaydet
    np.save(npy_file, np.array(df['overview_vectors'].tolist(), dtype=object))  # .npy dosyası


# Vektörleri yüklemek için fonksiyon
def load_overview_vectors():
    return pd.read_csv(vector_file)


# Filmlerin özetlerinden vektörler oluşturma
def get_overview_vectors(overviews_list):
    overview_vectors = []
    for overview in overviews_list:
        # Mevcut kelimelerin vektörlerini al
        vectors = [model.wv[word] for word in overview if word in model.wv]
        if len(vectors) > 0:
            # Özetlerin ortalama vektörünü hesapla
            overview_vectors.append(np.mean(vectors, axis=0))
        else:
            # Eğer vektör bulunamazsa, NaN döndür (boş geç)
            overview_vectors.append(np.nan)
    return overview_vectors


# Özet vektörlerini kaydetme veya yükleme
if not os.path.exists(vector_file):
    # processed_overview sütununu oluştur
    df['processed_overview'] = df['overview'].apply(lambda x: preprocess_text(x))

    # Vektörleri kaydet
    save_overview_vectors(df)
else:
    df_vectors = load_overview_vectors()
    df = df.merge(df_vectors, on='id')

