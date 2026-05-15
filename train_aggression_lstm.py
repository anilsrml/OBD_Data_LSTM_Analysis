from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FEATURE_COLS = [
    "Engine_RPM",
    "Vehicle_Speed",
    "Intake_MAP",
    "MAF",
    "Throttle_Pos",
    "Accel_Pedal_D",
]

# Modelin tahmin edeceği 3 agresiflik sınıfı.
CLASS_NAMES = {
    0: "calm",
    1: "normal",
    2: "aggressive",
}


def minmax(series: pd.Series) -> pd.Series:
    """Bir sütunu 0-1 aralığına ölçekler.

    Agresiflik skorunda farklı birimlerdeki değişkenleri birlikte kullanıyoruz.
    Bu yüzden RPM, hız, MAF gibi sütunların aynı ölçeğe getirilmesi gerekir.
    Eğer sütun sabitse bölme hatası oluşmaması için tüm değerler 0 yapılır.
    """
    min_value = series.min()
    max_value = series.max()
    value_range = max_value - min_value

    if value_range == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_value) / value_range


def load_data(csv_path: Path) -> pd.DataFrame:
    """İşlenmiş CSV dosyasını okur ve model için gerekli sütunları hazırlar.

    Bu fonksiyon:
    - CSV dosyasını pandas DataFrame olarak yükler.
    - Time ve model özellik sütunlarının varlığını kontrol eder.
    - Sayısal sütunları numeric tipe çevirir.
    - Eksik sayısal kayıtları çıkarır.
    """
    df = pd.read_csv(csv_path)

    missing_cols = [col for col in ["Time", *FEATURE_COLS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[["Time", *FEATURE_COLS]].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=FEATURE_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def add_rule_based_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Kural tabanlı agresiflik skoru ve 3 sınıflı etiket üretir.

    Veri setinde gerçek etiket olmadığı için agresiflik etiketi OBD
    sinyallerinden türetilir. Skor; pozitif hız değişimi, RPM, gaz pedalı,
    MAF ve manifold basıncını ağırlıklı şekilde birleştirir.

    Etiketler:
    0 -> sakin
    1 -> normal
    2 -> agresif
    """
    labeled_df = df.copy()

    # Hızdaki saniyelik değişim yaklaşık ivme bilgisini verir.
    labeled_df["Speed_Diff"] = labeled_df["Vehicle_Speed"].diff().fillna(0)

    # Sadece hızlanma agresiflik skoruna dahil edilir; yavaşlama negatif katkı yapmaz.
    labeled_df["Positive_Acceleration"] = labeled_df["Speed_Diff"].clip(lower=0)

    # Ağırlıklar sürüş agresifliği açısından en belirleyici sinyallere göre seçildi.
    # En yüksek ağırlık ani hızlanmaya verilir.
    labeled_df["Aggression_Score"] = (
        0.30 * minmax(labeled_df["Positive_Acceleration"])
        + 0.25 * minmax(labeled_df["Engine_RPM"])
        + 0.20 * minmax(labeled_df["Accel_Pedal_D"])
        + 0.15 * minmax(labeled_df["MAF"])
        + 0.10 * minmax(labeled_df["Intake_MAP"])
    )

    # qcut skoru 3 dengeli sınıfa böler. rank, aynı skorların sınırları bozmasını engeller.
    labeled_df["Aggression_Label"] = pd.qcut(
        labeled_df["Aggression_Score"].rank(method="first"),
        q=3,
        labels=[0, 1, 2],
    ).astype(int)

    return labeled_df


def create_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """LSTM için kayan pencere matrisleri oluşturur.

    Her örnek, ardışık `window_size` saniyelik veriden oluşur.
    Örneğin window_size=10 ise tek bir giriş örneği 10x6 matristir.
    Etiket olarak pencerenin son saniyesindeki agresiflik sınıfı alınır.

    Dönüş değerleri:
    - X: (örnek_sayısı, pencere_uzunluğu, özellik_sayısı)
    - y: Her pencereye karşılık gelen agresiflik etiketi
    - metadata: Pencere başlangıç/bitiş zamanları ve kural skoru
    """
    x_windows: list[np.ndarray] = []
    y_labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for start_idx in range(0, len(df) - window_size + 1, step_size):
        end_idx = start_idx + window_size - 1

        # LSTM girdisi: pencere içindeki 10 saniyelik özellik matrisi.
        x_windows.append(df.loc[start_idx:end_idx, FEATURE_COLS].to_numpy())

        # Hedef: pencerenin son anındaki agresiflik sınıfı.
        y_labels.append(int(df.loc[end_idx, "Aggression_Label"]))

        # Sonuçları sonradan yorumlamak için pencere bilgileri saklanır.
        metadata_rows.append(
            {
                "start_index": start_idx,
                "end_index": end_idx,
                "start_time": df.loc[start_idx, "Time"],
                "end_time": df.loc[end_idx, "Time"],
                "rule_label": int(df.loc[end_idx, "Aggression_Label"]),
                "rule_score": float(df.loc[end_idx, "Aggression_Score"]),
            }
        )

    return np.array(x_windows), np.array(y_labels), pd.DataFrame(metadata_rows)


def scale_windows(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """LSTM pencerelerini 0-1 aralığına ölçekler.

    Scaler sadece eğitim verisine fit edilir. Test verisi aynı scaler ile
    dönüştürülür. Böylece test bilgisinin eğitim aşamasına sızması engellenir.
    """
    n_features = x_train.shape[2]
    scaler = MinMaxScaler()

    # sklearn scaler 2 boyutlu veri beklediği için 3D LSTM verisi geçici olarak açılır.
    x_train_2d = x_train.reshape(-1, n_features)
    x_test_2d = x_test.reshape(-1, n_features)

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)

    return x_train_scaled, x_test_scaled, scaler


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """LSTM sınıflandırma modelini eğitir ve test tahminlerini üretir.

    Model, geçmiş pencere içindeki zamansal örüntüleri öğrenerek pencere
    sonundaki agresiflik sınıfını tahmin eder.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        ) from exc

    tf.random.set_seed(42)

    # Basit başlangıç mimarisi: LSTM -> Dropout -> Dense -> 3 sınıflı softmax.
    model = Sequential(
        [
            LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.30),
            Dense(32, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Doğrulama kaybı iyileşmezse eğitim erken durur ve en iyi ağırlıklar geri yüklenir.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return model, history, y_pred


def build_window_summary(x_windows: np.ndarray) -> np.ndarray:
    """Kümeleme için her pencereyi özet özelliklere çevirir.

    KMeans doğrudan 3 boyutlu LSTM girdisiyle çalışmaz. Bu yüzden her pencere
    ortalama, maksimum, standart sapma ve pencere başı-sonu farkı ile temsil edilir.
    """
    mean_values = x_windows.mean(axis=1)
    max_values = x_windows.max(axis=1)
    std_values = x_windows.std(axis=1)
    delta_values = x_windows[:, -1, :] - x_windows[:, 0, :]

    return np.concatenate([mean_values, max_values, std_values, delta_values], axis=1)


def run_clustering(
    x_windows: np.ndarray,
    y_labels: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """KMeans ile yan analiz yapar ve kural etiketleriyle karşılaştırır.

    Kümeleme doğrudan sınıf ismi üretmez. Bu nedenle kümeler, ortalama kural
    skorlarına göre düşükten yükseğe sıralanır ve 0/1/2 etiketlerine eşlenir.
    Adjusted Rand Score (ARI), kural tabanlı etiketlerle kümeleme etiketlerinin
    ne kadar uyumlu olduğunu özetler.
    """
    summary_features = build_window_summary(x_windows)
    scaled_summary = StandardScaler().fit_transform(summary_features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_summary)

    comparison = metadata.copy()
    comparison["cluster"] = clusters

    # Ortalama agresiflik skoru düşük olan küme sakin, yüksek olan küme agresif kabul edilir.
    cluster_order = (
        comparison.groupby("cluster")["rule_score"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    cluster_to_label = {cluster_id: label for label, cluster_id in enumerate(cluster_order)}
    comparison["cluster_label"] = comparison["cluster"].map(cluster_to_label).astype(int)

    ari = adjusted_rand_score(y_labels, comparison["cluster_label"])

    return comparison, ari


def save_json(path: Path, data: dict[str, object]) -> None:
    """Özet sonuç sözlüğünü okunabilir JSON dosyası olarak kaydeder."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    """Komut satırı akışını yönetir.

    Ana işlem sırası:
    1. CSV oku
    2. Kural tabanlı etiket üret
    3. Kayan pencereler oluştur
    4. Train/test ayır
    5. LSTM eğit
    6. KMeans ile yan analiz yap
    7. Tüm sonuçları output klasörüne kaydet
    """
    parser = argparse.ArgumentParser(
        description="Rule-based aggression labels, LSTM training, and clustering comparison."
    )
    parser.add_argument("--csv", default="data/df_1s_processed.csv", help="Input CSV path.")
    parser.add_argument("--output-dir", default="outputs/aggression_lstm", help="Output folder.")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding window size in seconds.")
    parser.add_argument("--step-size", type=int, default=1, help="Sliding window step in seconds.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum LSTM training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="LSTM batch size.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Time-based train split ratio.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSV oku ve kural tabanlı saniyelik etiketleri üret.
    df = load_data(csv_path)
    labeled_df = add_rule_based_labels(df)
    labeled_df.to_csv(output_dir / "df_labeled_rule_based.csv", index=False, encoding="utf-8-sig")

    # 2) Saniyelik veriyi LSTM'in beklediği kayan pencere formatına çevir.
    x_windows, y_labels, metadata = create_windows(
        labeled_df,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # 3) Zaman serisi olduğu için rastgele karıştırmadan sıralı train/test ayrımı yapılır.
    split_idx = int(len(x_windows) * args.train_ratio)
    x_train, x_test = x_windows[:split_idx], x_windows[split_idx:]
    y_train, y_test = y_labels[:split_idx], y_labels[split_idx:]
    test_metadata = metadata.iloc[split_idx:].reset_index(drop=True)

    # 4) Ölçekleyici eğitim verisine fit edilir ve daha sonra tekrar kullanım için kaydedilir.
    x_train_scaled, x_test_scaled, scaler = scale_windows(x_train, x_test)
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    # Çalışmanın genel ayarları ve sonuçları tek JSON raporda toplanır.
    report: dict[str, object] = {
        "input_csv": str(csv_path),
        "rows": int(len(df)),
        "window_size": args.window_size,
        "step_size": args.step_size,
        "features": FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "n_windows": int(len(x_windows)),
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        "label_distribution": labeled_df["Aggression_Label"].value_counts().sort_index().to_dict(),
    }

    try:
        # 5) TensorFlow kuruluysa LSTM eğitimi yapılır.
        model, history, y_pred = train_lstm(
            x_train_scaled,
            y_train,
            x_test_scaled,
            y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        model.save(output_dir / "aggression_lstm.keras")

        # Eğitim geçmişi loss/accuracy grafiklerini çizmek için ayrıca kaydedilir.
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(output_dir / "training_history.csv", index=False)

        # Test pencerelerinin gerçek ve tahmin edilen sınıfları dosyaya yazılır.
        test_predictions = test_metadata.copy()
        test_predictions["true_label"] = y_test
        test_predictions["pred_label"] = y_pred
        test_predictions["true_name"] = test_predictions["true_label"].map(CLASS_NAMES)
        test_predictions["pred_name"] = test_predictions["pred_label"].map(CLASS_NAMES)
        test_predictions.to_csv(output_dir / "lstm_test_predictions.csv", index=False)

        report["lstm"] = {
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=[CLASS_NAMES[i] for i in range(3)],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    except RuntimeError as exc:
        # TensorFlow yoksa script tamamen durmaz; etiketleme ve kümeleme sonuçları yine üretilir.
        report["lstm"] = {"skipped": True, "reason": str(exc)}
        print(str(exc))
        print("Continuing with clustering comparison only.")

    # 6) KMeans yan analizi ve kural tabanlı sınıflarla karşılaştırma.
    cluster_comparison, ari = run_clustering(x_windows, y_labels, metadata)
    cluster_comparison.to_csv(output_dir / "clustering_rule_comparison.csv", index=False)

    # Satırlar kural tabanlı etiketleri, sütunlar kümeleme etiketlerini gösterir.
    crosstab = pd.crosstab(
        cluster_comparison["rule_label"],
        cluster_comparison["cluster_label"],
        rownames=["rule_label"],
        colnames=["cluster_label"],
    )
    crosstab.to_csv(output_dir / "rule_vs_cluster_crosstab.csv")

    report["clustering"] = {
        "adjusted_rand_score": float(ari),
        "crosstab": crosstab.to_dict(),
    }

    # 7) Tüm özet metrikler JSON olarak kaydedilir.
    save_json(output_dir / "summary_report.json", report)

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"Labeled data: {output_dir / 'df_labeled_rule_based.csv'}")
    print(f"Summary report: {output_dir / 'summary_report.json'}")
    print(f"Rule vs cluster ARI: {ari:.4f}")


if __name__ == "__main__":
    main()
