# OBD Verisi ile LSTM Tabanlı Sürüş Agresifliği Analizi

## Genel Bakış

Bu proje, OBD zaman serisi verilerinden sürücünün sürüş agresifliğini tahmin etmeye yönelik bir makine öğrenmesi çalışmasıdır. Ham OBD kayıtları saniyelik forma dönüştürülür, modelde kullanılacak özellikler seçilir ve 10 saniyelik kayan pencereler üzerinden LSTM sınıflandırma modeli eğitilir.

Veri setinde manuel agresiflik etiketi bulunmadığı için etiketler kural tabanlı bir agresiflik skoru ile üretilir. KMeans kümeleme ise ana etiketleme yöntemi olarak değil, kural tabanlı etiketlerin verinin doğal kümelenme yapısıyla ne kadar örtüştüğünü inceleyen destekleyici analiz olarak kullanılır.

---

## Veri Kaynağı

Projede OBD verisi kullanılmaktadır. İşlenmiş ana veri dosyası:

```text
data/df_1s_processed.csv
```

Modelde kullanılan sütunlar:

| Sütun | Açıklama |
|---|---|
| `Engine_RPM` | Motor devri |
| `Vehicle_Speed` | Araç hızı |
| `Intake_MAP` | Emme manifoldu mutlak basıncı |
| `MAF` | Kütlesel hava akış oranı |
| `Throttle_Pos` | Gaz kelebeği konumu |
| `Accel_Pedal_D` | Gaz pedalı konumu |

---

## Proje Akışı

```text
OBD CSV
  │
  ▼
Saniyelik veri düzenleme
  │
  ▼
Kural tabanlı agresiflik skoru
  │
  ▼
3 sınıflı etiketleme
  │
  ▼
10 saniyelik kayan pencere
  │
  ▼
LSTM sınıflandırma modeli
  │
  ▼
KMeans ile yan analiz ve karşılaştırma
```

---

## Kural Tabanlı Etiketleme

Agresiflik skoru aşağıdaki sinyallerden türetilir:

- Pozitif hız değişimi
- Motor devri
- Gaz pedalı konumu
- Hava akış oranı
- Manifold basıncı

Skor daha sonra 3 sınıfa ayrılır:

| Etiket | Sınıf |
|---|---|
| `0` | Sakin |
| `1` | Normal |
| `2` | Agresif |

Bu yaklaşım, veri setinde gerçek sürücü etiketi olmadığı için pseudo-label üretme yöntemi olarak kullanılır.

---

## LSTM Modelleme

LSTM modeli, her örnek için 10 saniyelik geçmiş veriyi kullanır. Her pencere:

```text
10 zaman adımı x 6 özellik
```

boyutunda bir matristir. Model, pencerenin sonundaki agresiflik sınıfını tahmin etmek üzere eğitilir.

Zaman serisi yapısını korumak için train/test ayrımı rastgele değil, sıraya bağlı şekilde yapılır. Özellikler `MinMaxScaler` ile ölçeklenir ve eğitim sonunda model `.keras` formatında kaydedilir.

---

## KMeans Yan Analizi

KMeans, LSTM için etiket üretmez. Sadece kural tabanlı etiketlerin verideki doğal kümelenmeyle uyumunu değerlendirmek için kullanılır.

Karşılaştırma metrikleri:

- `Adjusted Rand Score`
- Kural etiketi / küme etiketi çapraz tablosu

---

## Ana Dosyalar

| Dosya | Açıklama |
|---|---|
| `train_aggression_lstm.py` | Etiketleme, pencereleme, LSTM eğitimi ve KMeans karşılaştırması |
| `lstm_analiz.ipynb` | Veri inceleme ve ön işleme notebook'u |
| `data/df_1s_processed.csv` | LSTM için hazırlanmış saniyelik veri |
| `README.md` | Kurulum ve çalıştırma bilgileri |

---

## Çıktılar

Script çıktıları `outputs/aggression_lstm/` altında üretilir. GitHub reposunda yalnızca aşağıdaki çıktı dosyaları paylaşılır:

- `training_history.csv`
- `summary_report.json`

Büyük ara çıktı dosyaları, tahmin dosyaları, model dosyaları ve scaler dosyaları `.gitignore` ile dışarıda bırakılır.

---

## Teknolojiler

- Python
- Pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- KMeans
- LSTM

---

*Bu belge, güncel proje yapısına göre hazırlanmıştır.*
