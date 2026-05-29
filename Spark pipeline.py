"""
=============================================================
  PROCESSING LAYER — Apache Spark / PySpark Pipeline
  Proyek: Platform Big Data Monitoring Gempa USGS
  Mata Kuliah: Mahadata / Big Data
=============================================================
Pipeline:
  Kafka/CSV → Spark Cleaning → Spark Analytics
    → Clustering (KMeans) → Spatio-Temporal Analysis
    → Hasil disimpan ke Parquet (pengganti HDFS) / HDFS
=============================================================
Output Big Data:
  - Throughput & Latency Spark
  - Distribusi magnitudo
  - Analisis temporal (per bulan/tahun)
  - Clustering wilayah rawan (KMeans via MLlib)
  - Statistik regional Indonesia
  - Heatmap data mentah (JSON untuk dashboard)
"""

import os
import time
import json
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────
#  COBA INIT SPARK (opsional — fallback ke pandas jika gagal)
# ──────────────────────────────────────────────────────────
SPARK_AVAILABLE = False
try:
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType
    )
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline as MLPipeline
    SPARK_AVAILABLE = True
except Exception as e:
    print(f"[WARN] PySpark tidak tersedia ({e}). Mode Pandas aktif.")

CSV_PATH      = 'query.csv'
OUTPUT_DIR    = 'spark_output'
PARQUET_PATH  = os.path.join(OUTPUT_DIR, 'cleaned_earthquake.parquet')
ANALYTICS_PATH = os.path.join(OUTPUT_DIR, 'analytics_results.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  A. MODE PYSPARK PENUH
# ══════════════════════════════════════════════════════════
def run_spark_pipeline():
    print("=" * 60)
    print("  PROCESSING LAYER — Apache Spark Pipeline")
    print("  Platform Big Data Monitoring Gempa USGS")
    print("=" * 60)

    # ── 1. Inisialisasi Spark Session ──────────────────────
    print("\n[1/6] Menginisialisasi Spark Session...")
    spark = (SparkSession.builder
             .appName("BigData_Gempa_USGS_Platform")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.memory", "2g")
             .config("spark.driver.host", "localhost")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    n_cores = spark.sparkContext.defaultParallelism
    print(f"      ✓ Spark aktif | Parallelism: {n_cores} core(s)")

    pipeline_start = time.time()

    # ── 2. Ingestion dari CSV (atau Kafka jika tersedia) ───
    print(f"\n[2/6] Membaca data dari: {CSV_PATH}")
    t0 = time.time()

    schema = StructType([
        StructField("time",      StringType(),  True),
        StructField("latitude",  DoubleType(),  True),
        StructField("longitude", DoubleType(),  True),
        StructField("depth",     DoubleType(),  True),
        StructField("mag",       DoubleType(),  True),
        StructField("magType",   StringType(),  True),
        StructField("nst",       DoubleType(),  True),
        StructField("gap",       DoubleType(),  True),
        StructField("dmin",      DoubleType(),  True),
        StructField("rms",       DoubleType(),  True),
        StructField("net",       StringType(),  True),
        StructField("id",        StringType(),  True),
        StructField("updated",   StringType(),  True),
        StructField("place",     StringType(),  True),
        StructField("type",      StringType(),  True),
        StructField("status",    StringType(),  True),
        StructField("locationSource", StringType(), True),
        StructField("magSource", StringType(),  True),
    ])

    raw_df = spark.read.csv(CSV_PATH, header=True, schema=schema)
    raw_count = raw_df.cache().count()
    ingestion_time = time.time() - t0
    print(f"      ✓ {raw_count:,} record dimuat dalam {ingestion_time:.3f} detik")

    # ── 3. Data Cleaning (Veracity) ────────────────────────
    print("\n[3/6] Data Cleaning & Veracity Handling...")
    t0 = time.time()

    cleaned_df = (raw_df
        .filter(F.col("type") == "earthquake")
        .filter(F.col("mag").isNotNull())
        .filter(F.col("latitude").isNotNull())
        .filter(F.col("longitude").isNotNull())
        .filter(F.col("depth").isNotNull())
        .filter(F.col("mag") >= 0)
        .filter(F.col("depth") >= 0)
        .filter(F.col("rms") <= 2.0)          # filter anomali kualitas sinyal
        .dropDuplicates(["time", "latitude", "longitude"])
    )

    # Tambahkan kolom turunan
    cleaned_df = (cleaned_df
        .withColumn("year",  F.year(F.to_timestamp("time")))
        .withColumn("month", F.month(F.to_timestamp("time")))
        .withColumn("hour",  F.hour(F.to_timestamp("time")))
        .withColumn("mag_category",
            F.when(F.col("mag") < 3.0, "Minor")
             .when(F.col("mag") < 5.0, "Light")
             .when(F.col("mag") < 6.0, "Moderate")
             .when(F.col("mag") < 7.0, "Strong")
             .otherwise("Major/Great"))
        .withColumn("depth_category",
            F.when(F.col("depth") <= 70,  "Dangkal (<70 km)")
             .when(F.col("depth") <= 300, "Menengah (70-300 km)")
             .otherwise("Dalam (>300 km)"))
    )

    clean_count = cleaned_df.cache().count()
    cleaning_time = time.time() - t0
    dropped = raw_count - clean_count

    print(f"      ✓ Raw: {raw_count:,} | Bersih: {clean_count:,} | Dibuang: {dropped:,}")
    print(f"      ✓ Veracity: {dropped/raw_count*100:.1f}% anomali dibuang")

    # ── 4. Analytics Layer ─────────────────────────────────
    print("\n[4/6] Menjalankan Spark Analytics...")
    t0 = time.time()

    # a. Distribusi magnitudo
    mag_dist = (cleaned_df.groupBy("mag_category")
                .agg(F.count("*").alias("jumlah"))
                .orderBy("jumlah", ascending=False)
                .toPandas())

    # b. Aktivitas per tahun-bulan
    temporal = (cleaned_df.groupBy("year", "month")
                .agg(F.count("*").alias("jumlah"),
                     F.avg("mag").alias("avg_mag"),
                     F.max("mag").alias("max_mag"))
                .orderBy("year", "month")
                .toPandas())

    # c. Wilayah paling aktif (grid 2°×2°)
    hotspot_df = (cleaned_df
        .withColumn("lat_grid",  F.round(F.col("latitude"),  0))
        .withColumn("lon_grid",  F.round(F.col("longitude"), 0))
        .groupBy("lat_grid", "lon_grid")
        .agg(F.count("*").alias("event_count"),
             F.avg("mag").alias("avg_mag"),
             F.max("mag").alias("max_mag"))
        .orderBy("event_count", ascending=False)
        .limit(200)
        .toPandas())

    # d. Statistik keseluruhan
    stats = cleaned_df.agg(
        F.count("*").alias("n"),
        F.avg("mag").alias("avg_mag"),
        F.max("mag").alias("max_mag"),
        F.min("mag").alias("min_mag"),
        F.stddev("mag").alias("std_mag"),
        F.avg("depth").alias("avg_depth"),
    ).collect()[0]

    analytics_time = time.time() - t0
    print(f"      ✓ Analytics selesai dalam {analytics_time:.3f} detik")

    # ── 5. Clustering KMeans (MLlib) ───────────────────────
    print("\n[5/6] Menjalankan KMeans Clustering (MLlib) ...")
    t0 = time.time()

    feature_cols = ["latitude", "longitude", "depth", "mag"]
    cluster_df = cleaned_df.select(feature_cols).dropna()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)
    kmeans    = KMeans(featuresCol="features", predictionCol="cluster",
                       k=6, seed=42, maxIter=20)

    ml_pipeline = MLPipeline(stages=[assembler, scaler, kmeans])
    model       = ml_pipeline.fit(cluster_df)
    clustered   = model.transform(cluster_df)

    cluster_summary = (clustered
        .groupBy("cluster")
        .agg(F.count("*").alias("jumlah_event"),
             F.avg("latitude").alias("center_lat"),
             F.avg("longitude").alias("center_lon"),
             F.avg("depth").alias("avg_depth"),
             F.avg("mag").alias("avg_mag"),
             F.max("mag").alias("max_mag"))
        .orderBy("cluster")
        .toPandas())

    cluster_time = time.time() - t0
    total_pipeline_time = time.time() - pipeline_start
    throughput = clean_count / total_pipeline_time

    print(f"      ✓ Clustering selesai dalam {cluster_time:.3f} detik")
    print(f"      ✓ 6 cluster zona seismik ditemukan")

    # ── 6. Simpan hasil ────────────────────────────────────
    print(f"\n[6/6] Menyimpan hasil ke '{OUTPUT_DIR}/'...")

    # Simpan cleaned data ke Parquet
    cleaned_df.write.mode("overwrite").parquet(PARQUET_PATH)

    # Simpan semua analytics ke JSON untuk dashboard
    results = {
        "pipeline_metrics": {
            "raw_records":       raw_count,
            "clean_records":     clean_count,
            "dropped_records":   dropped,
            "veracity_pct":      round(dropped / raw_count * 100, 2),
            "ingestion_time_sec": round(ingestion_time, 4),
            "cleaning_time_sec": round(cleaning_time, 4),
            "analytics_time_sec": round(analytics_time, 4),
            "clustering_time_sec": round(cluster_time, 4),
            "total_pipeline_latency_sec": round(total_pipeline_time, 4),
            "throughput_rec_per_sec": round(throughput, 2),
            "spark_cores": n_cores,
        },
        "global_stats": {
            "total_events": int(stats["n"]),
            "avg_magnitude": round(float(stats["avg_mag"]), 3),
            "max_magnitude": round(float(stats["max_mag"]), 3),
            "min_magnitude": round(float(stats["min_mag"]), 3),
            "std_magnitude": round(float(stats["std_mag"]), 3),
            "avg_depth_km": round(float(stats["avg_depth"]), 2),
        },
        "mag_distribution": mag_dist.to_dict(orient="records"),
        "temporal_activity": temporal.fillna(0).to_dict(orient="records"),
        "hotspot_grid": hotspot_df.fillna(0).to_dict(orient="records"),
        "cluster_summary": cluster_summary.fillna(0).to_dict(orient="records"),
    }

    with open(ANALYTICS_PATH, 'w') as f:
        json.dump(results, f, default=str)

    spark.stop()
    _print_metrics(results)
    return results


# ══════════════════════════════════════════════════════════
#  B. MODE PANDAS FALLBACK (tanpa Spark)
# ══════════════════════════════════════════════════════════
def run_pandas_pipeline():
    """Fallback saat PySpark tidak terinstal. Logika sama, output sama."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("  PROCESSING LAYER — Pandas Pipeline (Spark Fallback)")
    print("  Platform Big Data Monitoring Gempa USGS")
    print("=" * 60)

    pipeline_start = time.time()

    # 1. Load
    print(f"\n[1/6] Memuat dataset: {CSV_PATH}")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    raw_count = len(df)
    ingestion_time = time.time() - t0
    print(f"      ✓ {raw_count:,} record dalam {ingestion_time:.3f} detik")

    # 2. Cleaning
    print("\n[2/6] Data Cleaning...")
    t0 = time.time()
    df = df[df.get('type', pd.Series(['earthquake']*len(df))) == 'earthquake'] if 'type' in df.columns else df
    df = df.dropna(subset=['latitude', 'longitude', 'mag', 'depth'])
    df = df[df['mag'] >= 0]
    df = df[df['depth'] >= 0]
    if 'rms' in df.columns:
        df = df[df['rms'] <= 2.0]
    df = df.drop_duplicates(subset=['time', 'latitude', 'longitude']) if 'time' in df.columns else df.drop_duplicates()
    df = df.copy()

    # Parse waktu
    if 'time' in df.columns:
        df['time_parsed'] = pd.to_datetime(df['time'], errors='coerce')
        df['year']  = df['time_parsed'].dt.year
        df['month'] = df['time_parsed'].dt.month
        df['hour']  = df['time_parsed'].dt.hour

    # Kategori
    df['mag_category'] = pd.cut(df['mag'],
        bins=[-np.inf, 3.0, 5.0, 6.0, 7.0, np.inf],
        labels=['Minor', 'Light', 'Moderate', 'Strong', 'Major/Great'])
    df['depth_category'] = pd.cut(df['depth'],
        bins=[-np.inf, 70, 300, np.inf],
        labels=['Dangkal (<70 km)', 'Menengah (70-300 km)', 'Dalam (>300 km)'])

    clean_count = len(df)
    dropped = raw_count - clean_count
    cleaning_time = time.time() - t0
    print(f"      ✓ Bersih: {clean_count:,} | Dibuang: {dropped:,}")

    # 3. Analytics
    print("\n[3/6] Analytics...")
    t0 = time.time()

    mag_dist = (df.groupby('mag_category', observed=True)
                .size().reset_index(name='jumlah')
                .sort_values('jumlah', ascending=False))

    temporal = pd.DataFrame()
    if 'year' in df.columns:
        temporal = (df.groupby(['year', 'month'])
                    .agg(jumlah=('mag', 'count'),
                         avg_mag=('mag', 'mean'),
                         max_mag=('mag', 'max'))
                    .reset_index())

    df['lat_grid'] = df['latitude'].round(0)
    df['lon_grid'] = df['longitude'].round(0)
    hotspot_df = (df.groupby(['lat_grid', 'lon_grid'])
                  .agg(event_count=('mag', 'count'),
                       avg_mag=('mag', 'mean'),
                       max_mag=('mag', 'max'))
                  .reset_index()
                  .sort_values('event_count', ascending=False)
                  .head(200))

    analytics_time = time.time() - t0
    print(f"      ✓ Analytics selesai: {analytics_time:.3f} detik")

    # 4. Clustering KMeans
    print("\n[4/6] KMeans Clustering (6 cluster zona seismik)...")
    t0 = time.time()

    feature_cols = ['latitude', 'longitude', 'depth', 'mag']
    cluster_data = df[feature_cols].dropna().copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=100)
    cluster_data['cluster'] = kmeans.fit_predict(scaled)

    cluster_summary = (cluster_data.groupby('cluster')
                       .agg(jumlah_event=('mag', 'count'),
                            center_lat=('latitude', 'mean'),
                            center_lon=('longitude', 'mean'),
                            avg_depth=('depth', 'mean'),
                            avg_mag=('mag', 'mean'),
                            max_mag=('mag', 'max'))
                       .reset_index())

    cluster_time = time.time() - t0
    total_pipeline_time = time.time() - pipeline_start
    throughput = clean_count / total_pipeline_time

    print(f"      ✓ Clustering selesai: {cluster_time:.3f} detik")

    # 5. Simpan
    print(f"\n[5/6] Menyimpan hasil...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Simpan cleaned CSV & parquet-like (CSV untuk fallback)
    df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_earthquake.csv'), index=False)
    df[feature_cols + ['mag_category', 'depth_category',
                        'year', 'month'] if 'year' in df.columns
      else feature_cols + ['mag_category', 'depth_category']].head(50000).to_parquet(
          PARQUET_PATH, index=False) if hasattr(pd.DataFrame, 'to_parquet') else None

    results = {
        "pipeline_metrics": {
            "raw_records": raw_count,
            "clean_records": clean_count,
            "dropped_records": dropped,
            "veracity_pct": round(dropped / raw_count * 100, 2),
            "ingestion_time_sec": round(ingestion_time, 4),
            "cleaning_time_sec": round(cleaning_time, 4),
            "analytics_time_sec": round(analytics_time, 4),
            "clustering_time_sec": round(cluster_time, 4),
            "total_pipeline_latency_sec": round(total_pipeline_time, 4),
            "throughput_rec_per_sec": round(throughput, 2),
            "spark_cores": os.cpu_count() or 1,
            "mode": "Pandas (Spark Fallback)"
        },
        "global_stats": {
            "total_events": int(clean_count),
            "avg_magnitude": round(float(df['mag'].mean()), 3),
            "max_magnitude": round(float(df['mag'].max()), 3),
            "min_magnitude": round(float(df['mag'].min()), 3),
            "std_magnitude": round(float(df['mag'].std()), 3),
            "avg_depth_km": round(float(df['depth'].mean()), 2),
        },
        "mag_distribution": mag_dist.to_dict(orient="records"),
        "temporal_activity": temporal.fillna(0).to_dict(orient="records") if not temporal.empty else [],
        "hotspot_grid": hotspot_df.fillna(0).to_dict(orient="records"),
        "cluster_summary": cluster_summary.fillna(0).to_dict(orient="records"),
    }

    with open(ANALYTICS_PATH, 'w') as f:
        json.dump(results, f, default=str)

    _print_metrics(results)
    return results


# ══════════════════════════════════════════════════════════
#  HELPER — Cetak Tabel Metrik (output untuk laporan)
# ══════════════════════════════════════════════════════════
def _print_metrics(results: dict):
    m  = results["pipeline_metrics"]
    gs = results["global_stats"]

    print("\n" + "=" * 60)
    print("  OUTPUT PENELITIAN: PROCESSING LAYER (SPARK/PANDAS)")
    print("=" * 60)
    print(f"\n  ── VOLUME & VERACITY ──")
    print(f"  Total Data Mentah (Raw)        : {m['raw_records']:>12,} records")
    print(f"  Data Bersih Lolos Saring       : {m['clean_records']:>12,} records")
    print(f"  Anomali Dibuang (Veracity)     : {m['dropped_records']:>12,} records")
    print(f"  Persentase Data Bersih         : {100 - m['veracity_pct']:>11.2f}%")

    print(f"\n  ── PERFORMA PIPELINE ──")
    print(f"  Waktu Ingestion                : {m['ingestion_time_sec']:>11.4f} detik")
    print(f"  Waktu Cleaning                 : {m['cleaning_time_sec']:>11.4f} detik")
    print(f"  Waktu Analytics                : {m['analytics_time_sec']:>11.4f} detik")
    print(f"  Waktu Clustering               : {m['clustering_time_sec']:>11.4f} detik")
    print(f"  TOTAL LATENCY PIPELINE         : {m['total_pipeline_latency_sec']:>11.4f} detik")
    print(f"  THROUGHPUT SISTEM              : {m['throughput_rec_per_sec']:>10,.2f} records/detik")

    print(f"\n  ── STATISTIK GLOBAL GEMPA ──")
    print(f"  Total Event Seismik            : {gs['total_events']:>12,}")
    print(f"  Rata-Rata Magnitudo            : {gs['avg_magnitude']:>12.3f} SR")
    print(f"  Magnitudo Tertinggi            : {gs['max_magnitude']:>12.3f} SR")
    print(f"  Rata-Rata Kedalaman            : {gs['avg_depth_km']:>12.2f} km")

    print(f"\n  ── CLUSTERING ZONA SEISMIK ──")
    for c in results.get("cluster_summary", []):
        print(f"  Cluster {int(c['cluster'])}: {int(c['jumlah_event']):>7,} event | "
              f"center ({c['center_lat']:.1f}°, {c['center_lon']:.1f}°) | "
              f"avg mag {c['avg_mag']:.2f}")

    print("\n" + "=" * 60)
    print(f"  ✓ Hasil disimpan ke '{ANALYTICS_PATH}'")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    if SPARK_AVAILABLE:
        run_spark_pipeline()
    else:
        run_pandas_pipeline()