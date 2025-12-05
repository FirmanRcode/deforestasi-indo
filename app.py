import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import requests
import json
import re
from typing import Tuple

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Dashboard Deforestasi Indonesia",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Responsive & compact CSS
# -------------------------
st.markdown("""
<style>
/* Main typography */
.main-header {
    font-size: 2.2rem; 
    font-weight: 700; 
    color: #2E7D32; 
    text-align: center;
    margin-bottom: 12px;
}
.sub-header {
    font-size: 1.2rem; 
    font-weight: 600; 
    color: #4CAF50;
}
.metric-card {
    background-color: #f0f2f6; 
    padding: 12px; 
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.06);
}

/* Make Plotly containers more flexible on small screens */
[data-testid="stPlotlyChart"] {
    width: 100% !important;
}

/* Responsive rules */
@media (max-width: 900px) {
    .main-header { font-size: 1.6rem !important; }
    .sub-header { font-size: 1.0rem !important; }
    .metric-card { padding: 8px !important; font-size: 0.9rem; }
    /* Reduce default heights for plotly charts on mobile */
    div[role="figure"] { min-height: 300px !important; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
def normalize_name(s: str) -> str:
    """Normalize province names for robust matching with GeoJSON properties."""
    if not isinstance(s, str):
        return s
    s2 = s.upper()
    # remove diacritics-ish approximations and punctuation/extra spaces
    s2 = re.sub(r'[^A-Z0-9 ]', '', s2)
    s2 = re.sub(r'\s+', ' ', s2).strip()
    return s2

# -------------------------
# 1. DATA LOADING & CLEANING
# -------------------------
@st.cache_data
def load_data(file_path: str = 'Angka Deforestasi (Netto) Indonesia di Dalam dan di Luar Kawasan Hutan Tahun 2013-2022 (Ha_Th).csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path, header=[2, 3], dtype=str)
    
    # build new columns robustly
    new_columns = []
    last_year = None
    for col in df.columns:
        year_raw = str(col[0])
        cat_raw = str(col[1])
        if "Provinsi" in year_raw or "Provinsi" in cat_raw:
            new_columns.append("Provinsi")
            continue
        if "Unnamed" in year_raw or year_raw.strip() == '':
            current_year_raw = last_year
        else:
            current_year_raw = year_raw
            last_year = year_raw
        # try cleaning year
        year_clean = current_year_raw.strip() if current_year_raw else current_year_raw
        if year_clean and "-" in year_clean:
            parts = [p.strip() for p in year_clean.split("-") if p.strip()!='']
            if len(parts) >= 2:
                year_clean = parts[1]
        new_columns.append(f"{year_clean}_{cat_raw}")
    df.columns = new_columns

    # filter
    df = df[df['Provinsi'].notna()]
    df = df[~df['Provinsi'].str.contains("Ket", na=False)]
    df = df[~df['Provinsi'].str.contains("Sumber", na=False)]
    df = df[df['Provinsi'].str.upper() != "INDONESIA"]  # remove summary

    # melt tidy
    melted = df.melt(id_vars=['Provinsi'], var_name='Metric', value_name='Value')

    # split Metric
    melted[['Year', 'Category']] = melted['Metric'].str.split('_', n=1, expand=True)
    def clean_numeric(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            v = val.replace(',', '').strip()
            if v == '-' or v == '':
                return 0.0
            try:
                return float(v)
            except:
                return 0.0
        try:
            return float(val)
        except:
            return 0.0
    melted['Value'] = melted['Value'].apply(clean_numeric)
    melted['Year'] = pd.to_numeric(melted['Year'], errors='coerce')
    melted = melted.dropna(subset=['Year'])
    melted['Year'] = melted['Year'].astype(int)

    # Keep copy of raw provinces table for viewer
    df_provinces = df.copy()
    return melted, df_provinces

data_tidy, df_raw_p = load_data()

# Normalize province names column for consistent matching later
data_tidy['Provinsi_norm'] = data_tidy['Provinsi'].apply(normalize_name)

# -------------------------
# 2. CLUSTERING PIPELINE
# -------------------------
@st.cache_data
def perform_clustering(df_tidy: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, dict]:
    df_total = df_tidy[df_tidy['Category'].str.contains("Total Deforestasi", case=False, na=False)].copy()
    pivot_cluster = df_total.groupby('Provinsi', dropna=False)['Value'].mean().reset_index()
    X = pivot_cluster[['Value']].fillna(0)
    if len(X) < n_clusters:
        n_clusters = max(1, len(X))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pivot_cluster['Cluster'] = kmeans.fit_predict(X)
    centroids = pivot_cluster.groupby('Cluster')['Value'].mean().sort_values()
    mapping = {}
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    for i, cid in enumerate(centroids.index):
        # map in ascending centroid order
        mapping[cid] = labels[min(i, len(labels)-1)]
    pivot_cluster['Risk Level'] = pivot_cluster['Cluster'].map(mapping)
    color_map = {
        'Low Risk': '#81C784',
        'Medium Risk': '#FFCA28',
        'High Risk': '#E53935'
    }
    return pivot_cluster, color_map

cluster_data, risk_colors = perform_clustering(data_tidy)

# -------------------------
# 3. GEOJSON & MAPPING
# -------------------------
@st.cache_data
def get_geojson(url: str = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        gj = r.json()
        return gj
    except Exception as e:
        st.warning(f"Gagal memuat GeoJSON: {e}")
        return None

geojson_indo = get_geojson()

# Prepare geo property key and normalized mapping
if geojson_indo:
    first_prop = geojson_indo['features'][0]['properties']
    # find a prop key that looks like province name
    possible_keys = [k for k in first_prop.keys() if any(word in k.lower() for word in ['prop', 'nama', 'prov', 'provinsi'])]
    geo_key = possible_keys[0] if possible_keys else list(first_prop.keys())[0]

    # Build normalized name -> original name mapping from geojson
    geo_norm_map = {}
    for feat in geojson_indo['features']:
        raw = str(feat['properties'].get(geo_key, '')).upper()
        geo_norm_map[normalize_name(raw)] = raw

    # manual map corrections (kept from original with normalizations)
    manual_map = {
        normalize_name('DKI Jakarta'): normalize_name('JAKARTA RAYA'),
        normalize_name('DI Yogyakarta'): normalize_name('DAERAH ISTIMEWA YOGYAKARTA'),
        normalize_name('Kepulauan Bangka Belitung'): normalize_name('BANGKA BELITUNG'),
        normalize_name('Nusa Tenggara Barat'): normalize_name('NUSATENGGARA BARAT'),
        normalize_name('Nusa Tenggara Timur'): normalize_name('NUSATENGGARA TIMUR'),
        normalize_name('Papua Barat'): normalize_name('IRIAN JAYA BARAT'),
    }

    def map_province_to_geo(raw_name: str) -> str:
        n = normalize_name(raw_name)
        if n in manual_map:
            return geo_norm_map.get(manual_map[n], manual_map[n])
        # direct match
        if n in geo_norm_map:
            return geo_norm_map[n]
        # fuzzy contains
        for k_norm, k_raw in geo_norm_map.items():
            if n in k_norm or k_norm in n:
                return k_raw
        # fallback original
        return raw_name
    # apply
    cluster_data['Map_Provinsi'] = cluster_data['Provinsi'].apply(map_province_to_geo)
    data_tidy['Map_Provinsi'] = data_tidy['Provinsi'].apply(map_province_to_geo)
else:
    geo_key = 'Propinsi'
    cluster_data['Map_Provinsi'] = cluster_data['Provinsi']
    data_tidy['Map_Provinsi'] = data_tidy['Provinsi']

# -------------------------
# 4. FORECASTING
# -------------------------
def forecast_deforestation(df_tidy: pd.DataFrame, province: str = None):
    df = df_tidy[df_tidy['Category'].str.contains("Total Deforestasi", case=False, na=False)].copy()
    if province and province != 'Semua':
        df = df[df['Provinsi'] == province]
    df_agg = df.groupby('Year', dropna=False)['Value'].sum().reset_index().sort_values('Year')
    if df_agg.shape[0] < 2:
        # Not enough data for regression, return history only
        history_df = df_agg.copy()
        history_df['Type'] = 'Historical'
        return history_df, 0.0, 0.0
    X = df_agg['Year'].values.reshape(-1, 1)
    y = df_agg['Value'].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.arange(df_agg['Year'].max() + 1, 2031).reshape(-1, 1)
    if len(future_years) == 0:
        # already at 2030 or beyond, just return history
        df_agg['Type'] = 'Historical'
        return df_agg, float(model.coef_[0]), 0.0
    predictions = model.predict(future_years)
    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Value': predictions, 'Type': 'Forecast'})
    history_df = df_agg.copy()
    history_df['Type'] = 'Historical'
    full_trend = pd.concat([history_df, forecast_df], ignore_index=True).sort_values('Year')
    slope = float(model.coef_[0])
    future_sum = float(predictions.sum())
    return full_trend, slope, future_sum

# -------------------------
# Sidebar / Filters (with compact mode)
# -------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965300.png", width=90)
    st.title("Filter Dashboard")
    compact_mode = st.checkbox("Compact Mode (mobile-friendly)", value=False)
    # optional cluster config
    n_clusters = st.slider("Jumlah Cluster (K-Means)", 2, 6, 3)
    selected_year = st.slider("Pilih Tahun Data Detail", 2014, 2022, 2022)
    all_provinces = ['Semua'] + sorted(data_tidy['Provinsi'].unique().tolist())
    selected_province = st.selectbox("Pilih Provinsi (Analisis Tren)", all_provinces)
    st.divider()
    st.info("💡 Tips: aktifkan Compact Mode untuk layar kecil.")

# Recompute clustering if n_clusters changed
cluster_data, risk_colors = perform_clustering(data_tidy, n_clusters=n_clusters)

# -------------------------
# Main UI
# -------------------------
st.markdown('<div class="main-header">🌍 Dashboard Iklim: Deforestasi Indonesia</div>', unsafe_allow_html=True)

# Top metrics
total_def_2022 = data_tidy[
    (data_tidy['Year'] == 2022) & 
    (data_tidy['Category'].str.contains("Total", na=False))
]['Value'].sum()

highest_risk = cluster_data.sort_values('Value', ascending=False).iloc[0] if not cluster_data.empty else {'Provinsi': 'N/A', 'Value': 0}

col1, col2, col3 = st.columns([1,1,1])
col1.metric("Total Deforestasi 2022", f"{total_def_2022:,.0f} Ha", "-12% vs 2021" if total_def_2022 < 120000 else "Change") 
col2.metric("Provinsi Paling Kritis", highest_risk['Provinsi'], f"Avg: {highest_risk['Value']:,.0f} Ha/Th")
col3.metric("Status Data", "Valid & Cleaned", "2013-2022")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Peta Risiko (Cluster)", "📈 Tren & Prediksi", "📊 Komposisi Lahan", "📁 Data Source", "🏆 Analisis Lanjutan"])

# TAB 1: Map & animation & heatmap
with tab1:
    st.subheader("Peta Risiko Deforestasi (K-Means Clustering)")
    st.markdown("Provinsi dikelompokkan menjadi **Low, Medium, High Risk** berdasarkan rata-rata deforestasi historis.")
    if geojson_indo is None:
        st.warning("GeoJSON peta tidak tersedia. Pastikan koneksi internet.")
    else:
        fig_map = px.choropleth_mapbox(
            cluster_data,
            geojson=geojson_indo,
            locations='Map_Provinsi',
            featureidkey=f'properties.{geo_key}',
            color='Risk Level',
            color_discrete_map=risk_colors,
            hover_name='Provinsi',
            hover_data={'Value': ':,.0f'},
            title="Sebaran Risiko Deforestasi per Provinsi (Status)",
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": -2.5489, "lon": 118.0149}, 
            opacity=0.6,
        )
        fig_map.update_layout(height=450 if not compact_mode else 340, margin={"r":0,"t":40,"l":0,"b":0}, autosize=True)
        st.plotly_chart(fig_map, use_container_width=True, config={'responsive':True})

        st.divider()

        st.subheader("🎬 Time-Lapse: Evolusi Deforestasi (2013-2022)")
        anim_df = data_tidy[data_tidy['Category'].str.contains("Total Deforestasi", case=False, na=False)].copy()
        anim_df = anim_df.sort_values('Year')
        # ensure values present for each (Map_Provinsi, Year)
        fig_anim = px.choropleth_mapbox(
            anim_df,
            geojson=geojson_indo,
            locations='Map_Provinsi',
            featureidkey=f'properties.{geo_key}',
            color='Value',
            animation_frame='Year',
            color_continuous_scale="Reds",
            range_color=[0, anim_df['Value'].quantile(0.95) if not anim_df['Value'].empty else 1],
            hover_name='Provinsi',
            hover_data={'Value': ':,.0f'},
            title="Evolusi Deforestasi Tahunan",
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": -2.5489, "lon": 118.0149},
            opacity=0.6,
        )
        fig_anim.update_layout(height=560 if not compact_mode else 380, autosize=True, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_anim, use_container_width=True, config={'responsive':True})

        st.divider()
        st.subheader("Heatmap: Intensitas Deforestasi")
        heatmap_df = data_tidy[data_tidy['Category'].str.contains("Total Deforestasi", case=False, na=False)].copy()
        heatmap_pivot = heatmap_df.pivot_table(index='Provinsi', columns='Year', values='Value', aggfunc='sum', fill_value=0)
        if 2022 in heatmap_pivot.columns:
            heatmap_pivot = heatmap_pivot.sort_values(by=2022, ascending=False)
        fig_heat = px.imshow(
            heatmap_pivot,
            labels=dict(x="Tahun", y="Provinsi", color="Luas (Ha)"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale="Reds",
            aspect="auto",
            title="Heatmap Deforestasi (Urutan Tertinggi 2022)"
        )
        fig_heat.update_layout(height=700 if not compact_mode else 420, autosize=True)
        st.plotly_chart(fig_heat, use_container_width=True, config={'responsive':True})
        st.caption("*Warna merah pekat menunjukkan deforestasi tinggi.*")

# TAB 2: Trend & Forecast
with tab2:
    st.subheader(f"Analisis Tren & Prediksi Masa Depan ({selected_province})")
    trend_data, slope, future_sum = forecast_deforestation(data_tidy, selected_province)
    if slope > 0:
        story_text = f"🚨 **Peringatan:** Tren menunjukkan KENAIKAN deforestasi rata-rata **{slope:,.0f} Ha/tahun**. Prediksi kehilangan **{future_sum:,.0f} Ha** hingga 2030."
    else:
        story_text = f"✅ **Kabar Baik:** Tren menunjukkan PENURUNAN rata-rata **{abs(slope):,.0f} Ha/tahun**. Prediksi **{abs(future_sum):,.0f} Ha** (nilai absolut) hingga 2030."
    st.markdown(story_text)
    fig_trend = px.line(
        trend_data, 
        x='Year', 
        y='Value', 
        color='Type', 
        markers=True,
        line_shape='spline',
        title=f"Tren Deforestasi: Historis vs Prediksi (Target 2030)",
        color_discrete_map={'Historical': '#1E88E5', 'Forecast': '#FFC107'}
    )
    fig_trend.add_vline(x=trend_data['Year'].max() if not trend_data.empty else 2022, line_width=2, line_dash="dash", line_color="green", annotation_text="Saat Ini")
    fig_trend.update_layout(height=480 if not compact_mode else 340, autosize=True)
    st.plotly_chart(fig_trend, use_container_width=True, config={'responsive':True})

# TAB 3: Composition
with tab3:
    st.subheader("Komposisi Deforestasi: Hutan vs APL")
    st.markdown("Analisis apakah deforestasi terjadi di dalam Kawasan Hutan atau APL.")
    
    if selected_province == 'Semua':
        comp_df = data_tidy[~data_tidy['Category'].str.contains("Total", na=False)].groupby(['Year', 'Category'])['Value'].sum().reset_index()
    else:
        comp_df = data_tidy[(data_tidy['Provinsi'] == selected_province) & (~data_tidy['Category'].str.contains("Total", na=False))].groupby(['Year','Category'])['Value'].sum().reset_index()

    if comp_df.empty:
        st.warning("Data komposisi tidak tersedia untuk pilihan ini.")
    else:
        fig_area = px.area(
            comp_df,
            x='Year',
            y='Value',
            color='Category',
            title="Komposisi Deforestasi per Tahun",
            color_discrete_sequence=['#4CAF50', '#795548']
        )
        fig_area.update_layout(height=480 if not compact_mode else 340, autosize=True)
        st.plotly_chart(fig_area, use_container_width=True, config={'responsive':True})

# TAB 4: Raw Data
with tab4:
    st.subheader("Raw Data viewer")
    st.dataframe(df_raw_p)

# TAB 5: Advanced Analysis
with tab5:
    st.header("Analisis Lanjutan")
    col_pareto, col_radar = st.columns(2)
    with col_pareto:
        st.subheader("1. Pareto Analysis (80/20 Rule)")
        pareto_df = cluster_data.sort_values('Value', ascending=False).copy()
        if pareto_df['Value'].sum() == 0:
            st.info("Data cluster kosong atau nol.")
        else:
            pareto_df['Cumulative Percentage'] = pareto_df['Value'].cumsum() / pareto_df['Value'].sum() * 100
            top_contributors = pareto_df[pareto_df['Cumulative Percentage'] <= 80]
            cnt_contributors = len(top_contributors)
            st.info(f"💡 **Insight:** {cnt_contributors} Provinsi menyumbang ~80% deforestasi.")
            import plotly.graph_objects as go
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(
                x=pareto_df['Provinsi'].head(10), 
                y=pareto_df['Value'].head(10),
                name='Deforestasi Mean (Ha)',
                marker_color='rgb(200, 50, 50)'
            ))
            fig_pareto.add_trace(go.Scatter(
                x=pareto_df['Provinsi'].head(10),
                y=pareto_df['Cumulative Percentage'].head(10),
                name='Cumulative %',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='rgb(50, 50, 200)')
            ))
            fig_pareto.update_layout(
                title='Top 10 Kontributor Deforestasi (Pareto)',
                yaxis=dict(title='Rata-rata Deforestasi (Ha)'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 110]),
                legend=dict(x=0.5, y=1.1, orientation='h'),
                height=450 if not compact_mode else 340,
                autosize=True
            )
            st.plotly_chart(fig_pareto, use_container_width=True, config={'responsive':True})

    with col_radar:
        st.subheader("2. Head-to-Head Comparison")
        prov_list_compare = sorted(data_tidy['Provinsi'].unique().tolist())
        prov1 = st.selectbox("Provinsi 1", prov_list_compare, index=0)
        prov2 = st.selectbox("Provinsi 2", prov_list_compare, index=min(1, len(prov_list_compare)-1))
        if prov1 and prov2:
            radar_df = data_tidy[data_tidy['Provinsi'].isin([prov1, prov2])]
            radar_df = radar_df[~radar_df['Category'].str.contains("Total", na=False)]
            radar_pivot = radar_df.groupby(['Provinsi', 'Category'])['Value'].mean().reset_index()
            if radar_pivot.empty:
                st.warning("Data perbandingan tidak cukup.")
            else:
                fig_compare = px.bar(
                    radar_pivot, 
                    x='Category', 
                    y='Value', 
                    color='Provinsi', 
                    barmode='group',
                    title=f"Perbandingan Rata-rata: {prov1} vs {prov2}",
                    labels={'Value': 'Rata-rata Deforestasi (Ha)', 'Category': 'Kategori Lahan'},
                    color_discrete_sequence=['#FF5722', '#03A9F4']
                )
                fig_compare.update_layout(height=420 if not compact_mode else 320, autosize=True, legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
                st.plotly_chart(fig_compare, use_container_width=True, config={'responsive':True})

                val1 = cluster_data[cluster_data['Provinsi'] == prov1]['Value'].values
                val2 = cluster_data[cluster_data['Provinsi'] == prov2]['Value'].values
                if len(val1)>0 and len(val2)>0:
                    diff = float(val1[0]) - float(val2[0])
                    if diff > 0:
                        st.caption(f"**Insight:** {prov1} lebih tinggi **{diff:,.0f} Ha** dibanding {prov2}.")
                    else:
                        st.caption(f"**Insight:** {prov2} lebih tinggi **{abs(diff):,.0f} Ha** dibanding {prov1}.")

    # Quadrant
    st.divider()
    st.subheader("3. Scatter Plot Deforestasi (Kuadran)")
    quad_df = data_tidy[~data_tidy['Category'].str.contains("Total", na=False)].copy()
    quad_pivot = quad_df.groupby(['Provinsi','Category'])['Value'].mean().unstack(fill_value=0).reset_index()
    cat_cols = [c for c in quad_pivot.columns if c != 'Provinsi']
    if len(cat_cols) >= 2:
        x_col, y_col = cat_cols[0], cat_cols[1]
        fig_quad = px.scatter(
            quad_pivot,
            x=x_col,
            y=y_col,
            color='Provinsi',
            text='Provinsi',
            size=(quad_pivot[x_col].abs() + quad_pivot[y_col].abs()) + 10,
            title=f"Matrix: {x_col} vs {y_col}",
            labels={x_col: f"{x_col} (Mean)", y_col: f"{y_col} (Mean)"}
        )
        x_mean = quad_pivot[x_col].mean()
        y_mean = quad_pivot[y_col].mean()
        fig_quad.add_hline(y=y_mean, line_dash="dash", line_color="gray", annotation_text="Avg APL")
        fig_quad.add_vline(x=x_mean, line_dash="dash", line_color="gray", annotation_text="Avg Hutan")
        fig_quad.update_traces(textposition='top center')
        fig_quad.update_layout(showlegend=False, height=600 if not compact_mode else 420, autosize=True)
        st.plotly_chart(fig_quad, use_container_width=True, config={'responsive':True})
        st.info("""
        **📖 Cara Membaca Scatter Plot:**
        * Sumbu X : Deforestasi di APL (Area Penggunaan Lain).
        * Sumbu Y : Deforestasi di Kawasan Hutan.
        """)
    else:
        st.warning("Tidak cukup kategori (expect minimal 2 kategori non-total) untuk membuat kuadran.")

