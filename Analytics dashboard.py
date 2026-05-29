"""
=============================================================
  ANALYTICS & VISUALIZATION LAYER — Streamlit Dashboard
  Proyek: Platform Big Data Monitoring Gempa USGS
  Mata Kuliah: Mahadata / Big Data
=============================================================
Jalankan:
  streamlit run analytics_dashboard.py

Dashboard menampilkan output BIG DATA:
  ✓ Metrik pipeline (throughput, latency, volume)
  ✓ Distribusi magnitudo (5V characteristics)
  ✓ Aktivitas temporal (harian/bulanan/tahunan)
  ✓ Peta heatmap persebaran gempa
  ✓ Clustering zona seismik (KMeans 6 cluster)
  ✓ Analisis kedalaman vs magnitudo
  ✓ Wilayah hotspot paling aktif
  ✓ Forecasting tren sederhana (komponen opsional)
=============================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Platform Big Data Gempa USGS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Warna cluster
CLUSTER_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f",
    "#2ecc71", "#3498db", "#9b59b6"
]
CLUSTER_LABELS = [
    "Zona Merah (Sangat Aktif)",
    "Zona Oranye (Aktif)",
    "Zona Kuning (Moderat-Tinggi)",
    "Zona Hijau (Moderat)",
    "Zona Biru (Rendah-Moderat)",
    "Zona Ungu (Rendah)",
]

CSV_PATH       = 'query.csv'
ANALYTICS_PATH = 'spark_output/analytics_results.json'


# ══════════════════════════════════════════════════════════
#  HELPER: LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['latitude', 'longitude', 'mag', 'depth'])
    df = df[df['mag'] >= 0]
    df['time_parsed'] = pd.to_datetime(df.get('time', pd.Series()), errors='coerce')
    df['year']  = df['time_parsed'].dt.year
    df['month'] = df['time_parsed'].dt.month
    df['day']   = df['time_parsed'].dt.date
    df['mag_category'] = pd.cut(df['mag'],
        bins=[-np.inf, 2.0, 3.0, 5.0, 6.0, 7.0, np.inf],
        labels=['Micro (<2)', 'Minor (2-3)', 'Light (3-5)',
                'Moderate (5-6)', 'Strong (6-7)', 'Major+ (7+)'])
    df['depth_category'] = pd.cut(df['depth'],
        bins=[-np.inf, 70, 300, np.inf],
        labels=['Dangkal (<70 km)', 'Menengah (70-300 km)', 'Dalam (>300 km)'])
    return df


@st.cache_data
def load_analytics() -> dict:
    if os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def run_clustering(df: pd.DataFrame, n_clusters=6) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    cols = ['latitude', 'longitude', 'depth', 'mag']
    sample = df[cols].dropna().copy()
    if len(sample) > 100000:
        sample = sample.sample(100000, random_state=42)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=50)
    sample['cluster'] = km.fit_predict(scaled)
    return sample


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame):
    st.sidebar.image("https://earthquake.usgs.gov/theme/images/usgs-logo.svg",
                     width=140, use_container_width=False)
    st.sidebar.title("🌍 Filter & Kontrol")

    mag_range = st.sidebar.slider(
        "Rentang Magnitudo (SR)",
        float(df['mag'].min()), float(df['mag'].max()),
        (float(df['mag'].min()), float(df['mag'].max())),
        step=0.1
    )
    depth_range = st.sidebar.slider(
        "Rentang Kedalaman (km)",
        float(df['depth'].min()), min(float(df['depth'].max()), 700.0),
        (float(df['depth'].min()), min(float(df['depth'].max()), 700.0)),
        step=10.0
    )

    years = sorted(df['year'].dropna().unique().astype(int).tolist())
    if years:
        year_range = st.sidebar.select_slider(
            "Tahun",
            options=years,
            value=(min(years), max(years))
        )
    else:
        year_range = (None, None)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Platform Big Data Gempa USGS**")
    st.sidebar.markdown("Mata Kuliah: Mahadata / Big Data")

    return mag_range, depth_range, year_range


def apply_filters(df, mag_range, depth_range, year_range):
    mask = (
        (df['mag']   >= mag_range[0])   & (df['mag']   <= mag_range[1]) &
        (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1])
    )
    if year_range[0] and 'year' in df.columns:
        mask &= (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
    return df[mask].copy()


# ══════════════════════════════════════════════════════════
#  SECTION 1: HEADER & KPI
# ══════════════════════════════════════════════════════════
def render_header(df_filtered: pd.DataFrame, analytics: dict):
    st.markdown("""
    <h1 style='text-align:center; color:#e74c3c;'>
      🌍 Platform Big Data Monitoring Gempa Bumi USGS
    </h1>
    <p style='text-align:center; color:#666; font-size:16px;'>
      Analisis Spatio-Temporal Skala Besar | Apache Spark + Kafka | Real-Time Analytics
    </p>
    <hr/>
    """, unsafe_allow_html=True)

    m   = analytics.get("pipeline_metrics", {})
    gs  = analytics.get("global_stats", {})
    n   = len(df_filtered)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("📦 Total Record", f"{n:,}", help="Event gempa dalam filter aktif")
    with c2:
        st.metric("⚡ Throughput", f"{m.get('throughput_rec_per_sec', 0):,.0f} rec/s",
                  help="Record per detik yang diproses Spark")
    with c3:
        st.metric("⏱ Latency Pipeline", f"{m.get('total_pipeline_latency_sec', 0):.2f} s",
                  help="Total waktu pipeline end-to-end")
    with c4:
        st.metric("🔥 Mag. Tertinggi",
                  f"{df_filtered['mag'].max():.1f} SR" if n > 0 else "–")
    with c5:
        st.metric("🧹 Data Bersih",
                  f"{100 - m.get('veracity_pct', 0):.1f}%",
                  help="Persentase data lolos veracity check")
    with c6:
        st.metric("🗄 Raw Volume", f"{m.get('raw_records', n):,}",
                  help="Total record mentah sebelum cleaning")


# ══════════════════════════════════════════════════════════
#  SECTION 2: PIPELINE METRICS TABLE (Output Big Data)
# ══════════════════════════════════════════════════════════
def render_pipeline_metrics(analytics: dict):
    st.subheader("📊 Metrik Pipeline Big Data (Output Penelitian)")
    m = analytics.get("pipeline_metrics", {})
    if not m:
        st.info("Jalankan `spark_pipeline.py` terlebih dahulu untuk mengisi metrik ini.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Volume & Veracity**")
        df_v = pd.DataFrame([
            {"Metrik": "Total Data Mentah (Raw)", "Nilai": f"{m.get('raw_records', '-'):,} records"},
            {"Metrik": "Data Bersih (Lolos Filter)", "Nilai": f"{m.get('clean_records', '-'):,} records"},
            {"Metrik": "Anomali Dibuang (Veracity)", "Nilai": f"{m.get('dropped_records', '-'):,} records"},
            {"Metrik": "Persentase Data Valid", "Nilai": f"{100 - m.get('veracity_pct', 0):.2f}%"},
        ])
        st.dataframe(df_v, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Performa Sistem**")
        df_p = pd.DataFrame([
            {"Metrik": "Waktu Ingestion", "Nilai": f"{m.get('ingestion_time_sec', '-'):.4f} detik"},
            {"Metrik": "Waktu Cleaning / Veracity", "Nilai": f"{m.get('cleaning_time_sec', '-'):.4f} detik"},
            {"Metrik": "Waktu Analytics (Spark)", "Nilai": f"{m.get('analytics_time_sec', '-'):.4f} detik"},
            {"Metrik": "Waktu Clustering (MLlib)", "Nilai": f"{m.get('clustering_time_sec', '-'):.4f} detik"},
            {"Metrik": "TOTAL LATENCY PIPELINE", "Nilai": f"{m.get('total_pipeline_latency_sec', '-'):.4f} detik"},
            {"Metrik": "THROUGHPUT SISTEM", "Nilai": f"{m.get('throughput_rec_per_sec', '-'):,.2f} rec/detik"},
            {"Metrik": "Spark Parallelism (Cores)", "Nilai": str(m.get('spark_cores', '-'))},
        ])
        st.dataframe(df_p, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  SECTION 3: PETA HEATMAP
# ══════════════════════════════════════════════════════════
def render_heatmap(df: pd.DataFrame):
    st.subheader("🗺 Peta Persebaran & Heatmap Gempa")

    tab1, tab2 = st.tabs(["Heatmap Intensitas", "Scatter Magnitudo"])

    with tab1:
        sample = df.sample(min(len(df), 30000), random_state=42) if len(df) > 30000 else df
        fig = px.density_mapbox(
            sample, lat="latitude", lon="longitude",
            z="mag", radius=8,
            center={"lat": 0, "lon": 120},
            zoom=1.5,
            mapbox_style="carto-darkmatter",
            color_continuous_scale="YlOrRd",
            title="Heatmap Intensitas Seismik Global",
            labels={"mag": "Magnitudo"}
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sample = df.sample(min(len(df), 20000), random_state=1) if len(df) > 20000 else df
        fig2 = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude",
            color="mag", size="mag",
            color_continuous_scale="Inferno",
            size_max=12,
            zoom=1.5,
            mapbox_style="carto-positron",
            hover_data={"depth": True, "mag": True, "place": True}
                       if "place" in sample.columns else {"depth": True, "mag": True},
            title="Distribusi Spasial Gempa (ukuran = magnitudo)",
            labels={"mag": "Magnitudo"}
        )
        fig2.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  SECTION 4: TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════
def render_temporal(df: pd.DataFrame):
    st.subheader("📅 Analisis Temporal Aktivitas Seismik")

    tab1, tab2, tab3 = st.tabs(["Harian", "Bulanan", "Per Tahun"])

    with tab1:
        if 'day' in df.columns and df['day'].notna().any():
            daily = (df.groupby('day')
                     .agg(jumlah=('mag', 'count'), avg_mag=('mag', 'mean'))
                     .reset_index())
            daily['day'] = pd.to_datetime(daily['day'])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Jumlah Gempa per Hari", "Rata-Rata Magnitudo per Hari"))
            fig.add_trace(go.Bar(x=daily['day'], y=daily['jumlah'],
                                 name="Jumlah Event", marker_color="#e74c3c"), row=1, col=1)
            fig.add_trace(go.Scatter(x=daily['day'], y=daily['avg_mag'],
                                     mode='lines', name="Avg Mag", line=dict(color="#f39c12")), row=2, col=1)
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'month' in df.columns and 'year' in df.columns:
            monthly = (df.groupby(['year', 'month'])
                       .agg(jumlah=('mag', 'count'), avg_mag=('mag', 'mean'), max_mag=('mag', 'max'))
                       .reset_index())
            monthly['period'] = monthly.apply(
                lambda r: f"{int(r['year'])}-{int(r['month']):02d}" if pd.notna(r['year']) else '', axis=1)
            monthly = monthly.dropna(subset=['year']).sort_values('period')
            fig = px.bar(monthly, x='period', y='jumlah', color='avg_mag',
                         color_continuous_scale='Reds',
                         title="Jumlah Event Gempa per Bulan",
                         labels={"period": "Bulan", "jumlah": "Jumlah Event", "avg_mag": "Avg Mag"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'year' in df.columns:
            yearly = (df.groupby('year')
                      .agg(jumlah=('mag', 'count'), avg_mag=('mag', 'mean'), max_mag=('mag', 'max'))
                      .reset_index().dropna(subset=['year']))
            yearly['year'] = yearly['year'].astype(int)
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Event per Tahun", "Magnitudo Tertinggi per Tahun"))
            fig.add_trace(go.Bar(x=yearly['year'], y=yearly['jumlah'],
                                 marker_color="#3498db", name="Event"), row=1, col=1)
            fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['max_mag'],
                                     mode='lines+markers', line=dict(color="#e74c3c"),
                                     name="Max Mag"), row=1, col=2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  SECTION 5: DISTRIBUSI MAGNITUDO & KEDALAMAN
# ══════════════════════════════════════════════════════════
def render_distribution(df: pd.DataFrame):
    st.subheader("📈 Distribusi Magnitudo & Kedalaman")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='mag', nbins=50,
                           color_discrete_sequence=["#e74c3c"],
                           title="Distribusi Frekuensi Magnitudo",
                           labels={"mag": "Magnitudo (SR)", "count": "Frekuensi"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mag_cat = (df.groupby('mag_category', observed=True)
                   .size().reset_index(name='jumlah'))
        fig2 = px.pie(mag_cat, names='mag_category', values='jumlah',
                      title="Proporsi Kategori Magnitudo",
                      color_discrete_sequence=px.colors.sequential.RdBu_r)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        depth_cat = (df.groupby('depth_category', observed=True)
                     .size().reset_index(name='jumlah'))
        fig3 = px.bar(depth_cat, x='depth_category', y='jumlah',
                      color='jumlah', color_continuous_scale='Blues',
                      title="Distribusi Kedalaman Gempa",
                      labels={"depth_category": "Kategori Kedalaman", "jumlah": "Jumlah Event"})
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sample = df.sample(min(len(df), 10000), random_state=42)
        fig4 = px.scatter(sample, x='depth', y='mag',
                          color='mag_category', opacity=0.5,
                          title="Korelasi Kedalaman vs Magnitudo",
                          labels={"depth": "Kedalaman (km)", "mag": "Magnitudo (SR)"},
                          color_discrete_sequence=px.colors.qualitative.Set1)
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  SECTION 6: CLUSTERING ZONA SEISMIK
# ══════════════════════════════════════════════════════════
def render_clustering(df: pd.DataFrame):
    st.subheader("🔴 Clustering Zona Seismik (KMeans — 6 Cluster)")

    with st.spinner("Menjalankan KMeans Clustering..."):
        clustered = run_clustering(df)

    # Peta cluster
    clustered['cluster_label'] = clustered['cluster'].apply(
        lambda c: CLUSTER_LABELS[c] if c < len(CLUSTER_LABELS) else f"Cluster {c}")
    clustered['color'] = clustered['cluster'].apply(
        lambda c: CLUSTER_COLORS[c] if c < len(CLUSTER_COLORS) else "#aaa")

    fig = px.scatter_mapbox(
        clustered.sample(min(len(clustered), 30000), random_state=42),
        lat="latitude", lon="longitude",
        color="cluster_label",
        color_discrete_map={label: CLUSTER_COLORS[i] for i, label in enumerate(CLUSTER_LABELS)},
        zoom=1.5,
        mapbox_style="carto-darkmatter",
        title="Peta Clustering Zona Seismik (6 Zona Risiko)",
        opacity=0.6,
    )
    fig.update_layout(height=500, legend_title="Zona Risiko",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Tabel ringkasan cluster
    st.markdown("**Ringkasan Cluster Zona Seismik**")
    summary = (clustered.groupby('cluster')
               .agg(jumlah_event=('mag', 'count'),
                    center_lat=('latitude', 'mean'),
                    center_lon=('longitude', 'mean'),
                    avg_depth=('depth', 'mean'),
                    avg_mag=('mag', 'mean'),
                    max_mag=('mag', 'max'))
               .reset_index())
    summary['Zona'] = summary['cluster'].apply(
        lambda c: CLUSTER_LABELS[c] if c < len(CLUSTER_LABELS) else f"Cluster {c}")
    summary['Pusat'] = summary.apply(
        lambda r: f"{r['center_lat']:.1f}°, {r['center_lon']:.1f}°", axis=1)
    display = summary[['Zona', 'jumlah_event', 'Pusat', 'avg_depth', 'avg_mag', 'max_mag']].copy()
    display.columns = ['Zona', 'Jumlah Event', 'Pusat (Lat, Lon)', 'Avg Kedalaman (km)', 'Avg Mag', 'Max Mag']
    display['Avg Kedalaman (km)'] = display['Avg Kedalaman (km)'].round(1)
    display['Avg Mag'] = display['Avg Mag'].round(2)
    display['Max Mag'] = display['Max Mag'].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  SECTION 7: HOTSPOT TOP-20
# ══════════════════════════════════════════════════════════
def render_hotspot(df: pd.DataFrame):
    st.subheader("🔥 Top 20 Zona Hotspot Paling Aktif")

    df2 = df.copy()
    df2['lat_grid'] = df2['latitude'].round(1)
    df2['lon_grid'] = df2['longitude'].round(1)
    hotspot = (df2.groupby(['lat_grid', 'lon_grid'])
               .agg(event_count=('mag', 'count'),
                    avg_mag=('mag', 'mean'),
                    max_mag=('mag', 'max'))
               .reset_index()
               .sort_values('event_count', ascending=False)
               .head(20))

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(hotspot, x=hotspot.index, y='event_count',
                     color='avg_mag', color_continuous_scale='YlOrRd',
                     title="Top 20 Grid Lokasi Paling Aktif (resolusi 0.1°)",
                     labels={"event_count": "Jumlah Event", "avg_mag": "Avg Mag"})
        hotspot['label'] = hotspot.apply(
            lambda r: f"{r['lat_grid']:.1f}°, {r['lon_grid']:.1f}°", axis=1)
        fig.update_xaxes(tickvals=list(hotspot.index), ticktext=hotspot['label'], tickangle=45)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        disp = hotspot[['lat_grid', 'lon_grid', 'event_count', 'avg_mag', 'max_mag']].copy()
        disp.columns = ['Lat', 'Lon', 'Event', 'Avg Mag', 'Max Mag']
        disp['Avg Mag'] = disp['Avg Mag'].round(2)
        disp['Max Mag'] = disp['Max Mag'].round(2)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
#  SECTION 8: KARAKTERISTIK 5V BIG DATA
# ══════════════════════════════════════════════════════════
def render_5v(df: pd.DataFrame, analytics: dict):
    st.subheader("⚙ Karakteristik 5V Big Data pada Dataset Gempa USGS")
    m = analytics.get("pipeline_metrics", {})

    cols = st.columns(5)
    v_data = [
        ("📦 VOLUME", f"{len(df):,} records\nRaw: {m.get('raw_records', len(df)):,}",
         "Jutaan event seismik historis dari seluruh dunia"),
        ("⚡ VELOCITY", f"{m.get('throughput_rec_per_sec', 0):,.0f} rec/detik\nKafka ingestion real-time",
         "Data USGS diperbarui setiap menit via API & GeoJSON feed"),
        ("🌐 VARIETY", "CSV, JSON, GeoJSON\nParquet, Kafka Topic",
         "Multi-format: koordinat, magnitudo, fase, amplitudo, produk USGS"),
        ("✅ VERACITY", f"{100 - m.get('veracity_pct', 0):.1f}% valid\n{m.get('dropped_records', 0):,} anomali dibuang",
         "Filtering RMS, null, duplikat, dan nilai tak valid"),
        ("💡 VALUE", "Heatmap, Cluster\nEarly Warning, Tren",
         "Insight seismik untuk mitigasi bencana dan perencanaan wilayah"),
    ]

    for col, (title, value, desc) in zip(cols, v_data):
        col.markdown(f"""
        <div style='background:#1e1e2e; border:1px solid #e74c3c; border-radius:8px;
                    padding:14px; text-align:center; height:180px;'>
          <h3 style='color:#e74c3c; margin:0 0 8px 0; font-size:14px;'>{title}</h3>
          <p style='color:#fff; font-size:13px; font-weight:bold; white-space:pre-line;'>{value}</p>
          <p style='color:#aaa; font-size:11px; margin-top:8px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  SECTION 9: ARSITEKTUR PIPELINE
# ══════════════════════════════════════════════════════════
def render_architecture():
    st.subheader("🏗 Arsitektur Pipeline Big Data")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DATA SOURCE LAYER                             │
    │   USGS Earthquake API / CSV Feed (global, real-time, 5V)       │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                  INGESTION LAYER                                │
    │   Apache Kafka Producer → Topic 'gempa-stream'                  │
    │   Throughput: ribuan record/detik | Kompresi GZIP               │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │               STORAGE LAYER                                     │
    │   Kafka Consumer → Hadoop HDFS / Parquet Storage                │
    │   Format: Parquet (columnar) | Lokasi: /mahadata/kebencanaan/   │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │               PROCESSING LAYER (SPARK)                          │
    │   PySpark Cleaning → Transformasi → Agregasi                    │
    │   MLlib KMeans Clustering → Spatio-Temporal Analysis            │
    │   Latency: < 60 detik | Throughput: > 1000 rec/detik           │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │           ANALYTICS & VISUALIZATION LAYER                       │
    │   Streamlit Dashboard | Plotly Maps | Heatmap | Cluster         │
    │   Temporal Analysis | Hotspot Detection | 5V Metrics            │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)


# ══════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════
def main():
    # Load data
    if not os.path.exists(CSV_PATH):
        st.error(f"""
        ❌ File `{CSV_PATH}` tidak ditemukan!
        
        **Cara mendapatkan data:**
        1. Buka https://earthquake.usgs.gov/earthquakes/search/
        2. Set rentang tanggal (misal: 2010–2025)
        3. Klik **Search** → **Download** → pilih **CSV**
        4. Simpan file sebagai `query.csv` di folder yang sama dengan script ini
        5. Refresh halaman ini
        """)
        return

    df_raw = load_csv()
    analytics = load_analytics()

    # Sidebar filters
    mag_range, depth_range, year_range = render_sidebar(df_raw)
    df = apply_filters(df_raw, mag_range, depth_range, year_range)

    if len(df) == 0:
        st.warning("⚠ Tidak ada data yang cocok dengan filter yang dipilih.")
        return

    # Render sections
    render_header(df, analytics)
    st.markdown("---")

    render_pipeline_metrics(analytics)
    st.markdown("---")

    render_heatmap(df)
    st.markdown("---")

    render_temporal(df)
    st.markdown("---")

    render_distribution(df)
    st.markdown("---")

    render_clustering(df)
    st.markdown("---")

    render_hotspot(df)
    st.markdown("---")

    render_5v(df, analytics)
    st.markdown("---")

    render_architecture()

    st.markdown("---")
    st.markdown("""
    <p style='text-align:center; color:#666; font-size:12px;'>
    Platform Big Data Monitoring Gempa Bumi USGS &nbsp;|&nbsp;
    Apache Spark + Kafka + HDFS + Streamlit &nbsp;|&nbsp;
    Mata Kuliah Mahadata / Big Data
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()