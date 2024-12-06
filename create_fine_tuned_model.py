import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample
from tqdm import tqdm
import pandas as pd

# Modeli Yükleyin
model = SentenceTransformer('all-MiniLM-L6-v2')

# Veri setini yükleyin
df = pd.read_csv("tmdb_cleaned_no_revenue.csv")

# Eğitim verisini hazırlayın
train_examples = []

# Burada, veri kümesinde kaç örnek kullanılacağını belirtiriz
max_examples = 50000 # Örnek veri sayısını 10 ile sınırla

for i in range(min(max_examples, len(df) - 1)):
    for j in range(i + 1, min(i + 2, len(df))):  # Her örnek için sadece bir karşılaştırma yap
        similarity_score = 1.0 if df['genres'][i] == df['genres'][j] else 0.0
        train_examples.append(InputExample(texts=[df['overview'][i], df['overview'][j]], label=similarity_score))

# Dataset sınıfını oluşturun
class SentencePairDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

# Özel collate fonksiyonu
def collate_fn(batch):
    texts1 = [example.texts[0] for example in batch]
    texts2 = [example.texts[1] for example in batch]
    labels = torch.tensor([example.label for example in batch], dtype=torch.float)
    return texts1, texts2, labels

# Veri seti ve dataloader oluşturma
train_dataset = SentencePairDataset(train_examples)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)

# Eğitim ayarları
num_epochs = 4
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

# Modeli GPU'da çalıştırmak için, eğer GPU varsa kullanın
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Loss function
mse_loss = torch.nn.MSELoss()

# Eğitim döngüsü
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for texts1, texts2, labels in progress_bar:
        optimizer.zero_grad()

        # Metinleri encode et ve requires_grad=True olarak işaretle
        embeddings1 = model.encode(texts1, convert_to_tensor=True, device=device)
        embeddings2 = model.encode(texts2, convert_to_tensor=True, device=device)

        # Manually set requires_grad to True for the embeddings
        embeddings1.requires_grad = True
        embeddings2.requires_grad = True

        # Cosine similarity hesapla
        cosine_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

        # Etiketleri device'a taşı
        labels = labels.to(device)

        # Loss hesapla
        loss = mse_loss(cosine_scores, labels)

        # Geriye yayılım ve optimizasyon
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # İlerleme çubuğunu güncelle
        progress_bar.set_postfix({'loss': f"{total_loss / (progress_bar.n + 1):.4f}"})

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Modeli kaydetme
model.save("fine_tuned_model")
print("Fine-tuning tamamlandı ve model kaydedildi.")
