# LSTM Sürüş Agresifliği Analizi

Bu proje, `data/df_1s_processed.csv` verisini kullanarak kural tabanlı agresiflik etiketi üretir, 10 saniyelik kayan pencereler oluşturur, LSTM modeli eğitir ve KMeans ile yan kümeleme analizi yapar.

## Kurulum

Python 3.10+ önerilir.

```powershell
pip install pandas numpy scikit-learn joblib tensorflow
```

TensorFlow kurulmazsa script LSTM eğitimini atlar; yine de etiketleme ve kümeleme çıktıları üretilebilir.

## Script Akışı

`train_aggression_lstm.py` dosyası, işlenmiş OBD zaman serisini uçtan uca modelleme hattına dönüştürür. Önce `data/df_1s_processed.csv` okunur ve modelde kullanılacak özellikler seçilir: RPM, hız, manifold basıncı, MAF, gaz kelebeği ve pedal konumu.

Veri setinde manuel etiket olmadığı için kural tabanlı bir `Aggression_Score` hesaplanır. Bu skor; pozitif hız değişimi, motor devri, pedal konumu, hava akışı ve manifold basıncının normalize edilmiş ağırlıklı birleşimidir. Skor daha sonra 3 sınıfa ayrılır: sakin, normal ve agresif.

Etiketlenen veri 10 saniyelik kayan pencerelere çevrilir. Her pencere LSTM için `window_size x feature_count` boyutunda bir zaman serisi matrisi olur. Model, pencerenin sonundaki agresiflik sınıfını tahmin etmek üzere eğitilir.

Ek olarak KMeans, ana etiketleme yöntemi olarak değil, yan analiz için kullanılır. Amaç, kural tabanlı sınıfların verinin doğal kümelenme yapısıyla ne kadar örtüştüğünü `Adjusted Rand Score` ve çapraz tablo ile kontrol etmektir.

## Çalıştırma

Varsayılan ayarlarla:

```powershell
python train_aggression_lstm.py
```

Özel pencere ve epoch değeriyle:

```powershell
python train_aggression_lstm.py --window-size 10 --step-size 1 --epochs 30
```

## Çıktılar

Sonuçlar `outputs/aggression_lstm/` klasörüne kaydedilir:

- `df_labeled_rule_based.csv`: Kural tabanlı agresiflik skoru ve etiketler
- `aggression_lstm.keras`: Eğitilen LSTM modeli
- `lstm_test_predictions.csv`: Test tahminleri
- `clustering_rule_comparison.csv`: KMeans ve kural etiketi karşılaştırması
- `summary_report.json`: Özet metrikler

## Sınıflar

- `0`: sakin
- `1`: normal
- `2`: agresif
