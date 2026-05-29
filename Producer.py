"""
=============================================================
  INGESTION LAYER — Kafka Producer
  Proyek: Platform Big Data Monitoring Gempa USGS
  Mata Kuliah: Mahadata / Big Data
=============================================================
Fungsi:
  Membaca dataset gempa bumi dari USGS (CSV),
  lalu mengirimkan setiap record ke Kafka Topic 'gempa-stream'
  secara bulk. Mengukur throughput & latency ingestion.
"""

import time
import json
import pandas as pd

# ──────────────────────────────────────────────────────────
#  COBA IMPORT KAFKA (opsional — jika tidak ada, simulasikan)
# ──────────────────────────────────────────────────────────
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("[WARN] kafka-python tidak ditemukan. Mode SIMULASI aktif.")

KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME   = 'gempa-stream'
CSV_PATH     = 'query.csv'          # ganti dengan nama file CSV USGS kalian

# ──────────────────────────────────────────────────────────
#  FUNGSI UTAMA
# ──────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Muat CSV USGS dan normalisasi kolom."""
    df = pd.read_csv(path)
    # USGS mungkin punya nama kolom berbeda — pastikan mapping ini
    rename_map = {}
    if 'time'      not in df.columns and 'Time'      in df.columns: rename_map['Time']      = 'time'
    if 'latitude'  not in df.columns and 'Latitude'  in df.columns: rename_map['Latitude']  = 'latitude'
    if 'longitude' not in df.columns and 'Longitude' in df.columns: rename_map['Longitude'] = 'longitude'
    if 'depth'     not in df.columns and 'Depth'     in df.columns: rename_map['Depth']     = 'depth'
    if 'mag'       not in df.columns and 'Magnitude' in df.columns: rename_map['Magnitude'] = 'mag'
    df = df.rename(columns=rename_map)

    required = ['time', 'latitude', 'longitude', 'depth', 'mag']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di CSV. Kolom tersedia: {list(df.columns)}")
    return df


def build_record(row: pd.Series) -> dict:
    """Bangun payload JSON yang akan dikirim ke Kafka."""
    return {
        'time':      str(row.get('time', '')),
        'latitude':  float(row['latitude']),
        'longitude': float(row['longitude']),
        'depth':     float(row['depth'])       if pd.notnull(row.get('depth'))  else 0.0,
        'mag':       float(row['mag'])         if pd.notnull(row.get('mag'))    else 0.0,
        'magType':   str(row.get('magType', 'unknown')),
        'place':     str(row.get('place', 'unknown')),
        'type':      str(row.get('type', 'earthquake')),
        'nst':       float(row['nst'])         if pd.notnull(row.get('nst'))    else 0.0,
        'gap':       float(row['gap'])         if pd.notnull(row.get('gap'))    else 0.0,
        'rms':       float(row['rms'])         if pd.notnull(row.get('rms'))    else 0.0,
    }


def run_producer():
    print("=" * 55)
    print("  INGESTION LAYER — Kafka Producer")
    print("  Platform Big Data Monitoring Gempa Bumi USGS")
    print("=" * 55)

    # 1. Load dataset
    print(f"\n[1/4] Memuat dataset dari: {CSV_PATH}")
    df = load_dataset(CSV_PATH)
    total_records = len(df)
    print(f"      ✓ {total_records:,} record berhasil dimuat")

    # 2. Inisialisasi producer
    print(f"\n[2/4] Menghubungkan ke Kafka Broker ({KAFKA_BROKER})...")
    if KAFKA_AVAILABLE:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=65536,          # 64 KB batch untuk throughput tinggi
                linger_ms=10,             # tunggu 10 ms agar batch lebih penuh
                compression_type='gzip',  # kompresi untuk efisiensi jaringan
            )
            print("      ✓ Terhubung ke Kafka")
        except Exception as e:
            print(f"      ✗ Gagal terhubung ke Kafka: {e}")
            print("      → Beralih ke mode SIMULASI")
            KAFKA_AVAILABLE_LOCAL = False
            producer = None
    else:
        KAFKA_AVAILABLE_LOCAL = False
        producer = None

    # 3. Kirim data
    print(f"\n[3/4] Memulai BULK INGESTION ke topic '{TOPIC_NAME}'...")
    start_time = time.time()
    success_count = 0
    fail_count    = 0
    simulated_buffer = []   # untuk mode simulasi

    for idx, row in df.iterrows():
        try:
            record = build_record(row)
            if producer:
                producer.send(TOPIC_NAME, value=record)
            else:
                simulated_buffer.append(record)   # simpan di memori (simulasi)
            success_count += 1
        except Exception as e:
            fail_count += 1

        # Progress log setiap 10.000 record
        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            tput = success_count / elapsed if elapsed > 0 else 0
            print(f"      Progress: {success_count:,}/{total_records:,} record | {tput:,.0f} rec/detik")

    if producer:
        producer.flush()

    end_time  = time.time()
    duration  = end_time - start_time
    throughput = success_count / duration if duration > 0 else 0
    data_loss_pct = (fail_count / total_records * 100) if total_records > 0 else 0

    # Hitung estimasi ukuran data (asumsi ~200 byte per record JSON)
    est_bytes = success_count * 200
    est_mb    = est_bytes / (1024 * 1024)

    # 4. Cetak metrik output Big Data
    print("\n" + "=" * 55)
    print("  OUTPUT PENELITIAN: INGESTION LAYER (KAFKA)")
    print("=" * 55)
    print(f"  Total Record (Volume)         : {total_records:>12,} records")
    print(f"  Record Berhasil Dikirim       : {success_count:>12,} records")
    print(f"  Record Gagal / Error          : {fail_count:>12,} records")
    print(f"  Data Loss                     : {data_loss_pct:>11.2f}%")
    print(f"  Durasi Ingestion              : {duration:>11.4f} detik")
    print(f"  THROUGHPUT SISTEM             : {throughput:>10,.2f} records/detik")
    print(f"  Est. Volume Data Dikirim      : {est_mb:>11.2f} MB")
    print(f"  Kompresi                      :        GZIP aktif")
    print(f"  Kafka Topic                   :    {TOPIC_NAME}")
    print(f"  Mode                          : {'KAFKA REAL' if producer else 'SIMULASI'}")
    print("=" * 55)

    # Simpan hasil ke file untuk laporan
    result = {
        'total_records': total_records,
        'success_count': success_count,
        'fail_count': fail_count,
        'data_loss_pct': data_loss_pct,
        'duration_sec': round(duration, 4),
        'throughput_rec_per_sec': round(throughput, 2),
        'estimated_mb': round(est_mb, 2),
    }
    import json
    with open('ingestion_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Metrik disimpan ke 'ingestion_metrics.json'")

    # Kalau simulasi, simpan buffer ke file JSON lokal (pengganti Kafka)
    if not producer and simulated_buffer:
        with open('simulated_kafka_buffer.json', 'w') as f:
            json.dump(simulated_buffer, f)
        print(f"  ✓ Buffer simulasi disimpan ke 'simulated_kafka_buffer.json'")

    return result


if __name__ == '__main__':
    run_producer()