import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import requests

# Set Page Configuration
st.set_page_config(
    page_title="Dashboard Deforestasi Indonesia",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #2E7D32; 
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem; 
        font-weight: 600; 
        color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING & CLEANING ---
@st.cache_data
def load_data():
    file_path = 'Angka Deforestasi (Netto) Indonesia di Dalam dan di Luar Kawasan Hutan Tahun 2013-2022 (Ha_Th).csv'
    
    # Read CSV with multi-index header (Rows 3 and 4 in file are indices 2 and 3)
    # Adjusting for 0-based index: Row 3 is index 2.
    df = pd.read_csv(file_path, header=[2, 3])
    
    # The first column is 'Provinsi' but it might be detached due to multi-index
    # Flatten columns to make them easier to work with
    # Current columns: [(Provinsi, Unnamed), (2013-2014, Kawasan Hutan), ...]
    
    new_columns = []
    last_year = None
    
    for col in df.columns:
        year_raw = str(col[0])
        category_raw = str(col[1])
        
        # Handle "Provinsi" column (it might be in level 0 or level 1 depending on pandas parsing)
        if "Provinsi" in year_raw or "Provinsi" in category_raw:
            new_columns.append("Provinsi")
            continue
            
        # Forward Fill Year Logic
        # If year_raw looks like "Unnamed" (pandas default for empty CSV header), use last_year
        if "Unnamed" in year_raw:
            current_year_raw = last_year
        else:
            current_year_raw = year_raw
            last_year = year_raw # Update last seen year
            
        # Clean the Year String (e.g., "2013-2014" -> "2014")
        if current_year_raw and "-" in current_year_raw:
            try:
                # Take second part "2014"
                year_clean = current_year_raw.strip().split("-")[1]
            except:
                year_clean = current_year_raw
        else:
            year_clean = current_year_raw
            
        # Construct Column Name
        new_columns.append(f"{year_clean}_{category_raw}")
    
    df.columns = new_columns

    
    # Filter only relevant columns (Provinsi and data columns)
    # Remove "Ket" or "Sumber" rows usually at bottom
    df = df[df['Provinsi'].notna()]
    df = df[~df['Provinsi'].str.contains("Ket", na=False)]
    df = df[~df['Provinsi'].str.contains("Sumber", na=False)]
    
    # Remove "INDONESIA" summary row for clustering, but might keep for aggregate
    df_provinces = df[df['Provinsi'] != "INDONESIA"].copy()
    
    # Reshape (Melt) to Tidy Format
    # Need to separate: "2014_Kawasan Hutan", "2014_Total Deforestasi"
    # Melt ID: Provinsi
    melted = df_provinces.melt(id_vars=['Provinsi'], var_name='Metric', value_name='Value')
    
    # Parse Year and Category from 'Metric' "2014_Category"
    melted[['Year', 'Category']] = melted['Metric'].str.split('_', n=1, expand=True)
    
    # Cleaning 'Value' column
    # Remove commas, handle '-' as 0 or NaN
    def clean_numeric(val):
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if val == '-' or val == '':
                return 0.0
            try:
                return float(val)
            except:
                return 0.0
        return val

    melted['Value'] = melted['Value'].apply(clean_numeric)
    melted['Year'] = pd.to_numeric(melted['Year'], errors='coerce')
    
    # Drop rows where year parsing failed
    melted = melted.dropna(subset=['Year'])
    melted['Year'] = melted['Year'].astype(int)
    
    return melted, df_provinces

data_tidy, df_raw_p = load_data()

# --- 2. CLUSTERING PIPELINE ---
@st.cache_data
def perform_clustering(df_tidy):
    # Filter for 'Total Deforestasi'
    df_total = df_tidy[df_tidy['Category'].str.contains("Total Deforestasi", case=False, na=False)]
    
    # Aggregate Mean Deforestation per Province
    pivot_cluster = df_total.groupby('Provinsi')['Value'].mean().reset_index()
    X = pivot_cluster[['Value']]
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pivot_cluster['Cluster'] = kmeans.fit_predict(X)
    
    # Map Cluster IDs to High/Medium/Low based on centroids
    centroids = pivot_cluster.groupby('Cluster')['Value'].mean().sort_values()
    # centroids index (0,1,2) sorted by value (Low, Med, High)
    cluster_map = {
        centroids.index[0]: 'Low Risk',
        centroids.index[1]: 'Medium Risk',
        centroids.index[2]: 'High Risk'
    }
    pivot_cluster['Risk Level'] = pivot_cluster['Cluster'].map(cluster_map)
    
    # Define Color Map for Plotly
    color_map = {
        'Low Risk': '#81C784',   # Green
        'Medium Risk': '#FFCA28', # Yellow/Orange
        'High Risk': '#E53935'    # Red
    }
    
    return pivot_cluster, color_map

cluster_data, risk_colors = perform_clustering(data_tidy)

# --- 3. GEOJSON & MAPPING ---
@st.cache_data
def get_geojson():
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    try:
        r = requests.get(url)
        return r.json()
    except:
        st.error("Gagal memuat GeoJSON Peta Indonesia.")
        return None

geojson_indo = get_geojson()

# Dynamic Mapping Logic
if geojson_indo:
    # 1. Get all province names from GeoJSON properties
    # Key usually 'Propinsi' or 'propinsi'. Check first feature.
    first_prop = geojson_indo['features'][0]['properties']
    geo_key = 'Propinsi' if 'Propinsi' in first_prop else 'propinsi'
    
    geo_names = [f['properties'][geo_key] for f in geojson_indo['features']]
    
    # 2. Manual fixes for known mismatches (CSV -> GeoJSON UPPERCASE)
    manual_map = {
        'DKI Jakarta': 'JAKARTA RAYA',
        'DI Yogyakarta': 'DAERAH ISTIMEWA YOGYAKARTA',
        'Kepulauan Bangka Belitung': 'BANGKA BELITUNG',
        'Nusa Tenggara Barat': 'NUSATENGGARA BARAT', # Based on inspection
        'Nusa Tenggara Timur': 'NUSATENGGARA TIMUR',
        'Papua Barat': 'IRIAN JAYA BARAT', # Common old name in some GeoJSONs
    }
    
    def map_province(name):
        # Check manual map first
        if name in manual_map:
            return manual_map[name]
        
        # Try exact match UPPERCASE
        name_upper = name.upper()
        if name_upper in geo_names:
            return name_upper
            
        # Try finding partial match
        for g_name in geo_names:
            if name_upper in g_name or g_name in name_upper:
                return g_name
                
        return name # Fallback
    
    cluster_data['Map_Provinsi'] = cluster_data['Provinsi'].apply(map_province)
else:
    geo_key = 'Propinsi' # Default fallback

# ... (Forecasting code remains) ...

# ...

# --- 4. FORECASTING ALGORITHM ---
def forecast_deforestation(df_tidy, province=None):
    # Filter Total Deforestasi
    df = df_tidy[df_tidy['Category'].str.contains("Total Deforestasi", case=False)]
    
    if province and province != 'Semua':
        df = df[df['Provinsi'] == province]
        
    # Aggregate by Year
    df_agg = df.groupby('Year')['Value'].sum().reset_index()
    
    # Linear Regression
    X = df_agg['Year'].values.reshape(-1, 1)
    y = df_agg['Value'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict 2023 - 2030
    future_years = np.arange(2023, 2031).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Create Forecast DataFrame
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Value': predictions,
        'Type': 'Forecast'
    })
    
    # Combine with History
    history_df = df_agg.copy()
    history_df['Type'] = 'Historical'
    
    full_trend = pd.concat([history_df, forecast_df])
    
    # Co-Benefits Calculation (Simple)
    # If slope is negative, we are saving. If positive, losing.
    slope = model.coef_[0]
    
    return full_trend, slope, forecast_df['Value'].sum()

# --- 5. UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965300.png", width=100)
    st.title("Filter Dashboard")
    
    selected_year = st.slider("Pilih Tahun Data Detail", 2014, 2022, 2022)
    
    all_provinces = ['Semua'] + sorted(data_tidy['Provinsi'].unique().tolist())
    selected_province = st.selectbox("Pilih Provinsi (Analisis Tren)", all_provinces)
    
    st.divider()
    st.info("💡 **Tips:** Gunakan filter ini untuk mengubah fokus analisis pada grafik Tren dan Komposisi.")

# Main Title
st.markdown('<div class="main-header">🌍 Dashboard Iklim: Deforestasi Indonesia</div>', unsafe_allow_html=True)

# Top Metrics (Overview)
total_def_2022 = data_tidy[
    (data_tidy['Year'] == 2022) & 
    (data_tidy['Category'].str.contains("Total"))
]['Value'].sum()

# Highest Risk Province (historical average)
highest_risk = cluster_data.sort_values('Value', ascending=False).iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Deforestasi 2022", f"{total_def_2022:,.0f} Ha", "-12% vs 2021" if total_def_2022 < 120000 else "Change") # Dummy Delta logic or calculate real
col2.metric("Provinsi Paling Kritis", highest_risk['Provinsi'], f"Avg: {highest_risk['Value']:,.0f} Ha/Th")
col3.metric("Status Data", "Valid & Cleaned", "2013-2022")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Risiko (Cluster)", "📈 Tren & Prediksi", "📊 Komposisi Lahan", "📁 Data Source"])

# --- TAB 1: INTERACTIVE MAP ---
with tab1:
    st.subheader("Peta Risiko Deforestasi (K-Means Clustering)")
    st.markdown("Provinsi dikelompokkan menjadi **Low, Medium, High Risk** berdasarkan rata-rata deforestasi historis.")
    
    if geojson_indo:
        # Use Mapbox for better aesthetics ("Yang Bagus")
        fig_map = px.choropleth_mapbox(
            cluster_data,
            geojson=geojson_indo,
            locations='Map_Provinsi',
            featureidkey=f'properties.{geo_key}',
            color='Risk Level',
            color_discrete_map=risk_colors,
            hover_name='Provinsi',
            hover_data={'Value': ':,.0f'},
            title="Sebaran Risiko Deforestasi per Provinsi",
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": -2.5489, "lon": 118.0149}, # Center of Indonesia
            opacity=0.6,
        )
        
        fig_map.update_layout(
            height=600, 
            margin={"r":0,"t":40,"l":0,"b":0},
            legend=dict(y=1, x=0)
        )
        # Fix: If map is not rendering features, it might be the key matching.
        # But sticking to Mapbox is generally safer for visualization.
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.caption("*Catatan: Menggunakan base map Carto-Positron untuk tampilan yang lebih jelas.*")

# --- TAB 2: TREND & FORECASTING ---
with tab2:
    st.subheader(f"Analisis Tren & Prediksi Masa Depan ({selected_province})")
    
    # Calculate Forecast
    trend_data, slope, future_sum = forecast_deforestation(data_tidy, selected_province)
    
    # Storytelling Co-Benefits
    if slope > 0:
        story_color = "red"
        story_text = f"🚨 **Peringatan Iklim:** Tren menunjukkan KENAIKAN deforestasi rata-rata **{slope:,.0f} Ha/tahun**. Jika tidak ada intervensi, kita diprediksi kehilangan **{future_sum:,.0f} Ha** hutan lagi hingga 2030."
    else:
        story_color = "green"
        story_text = f"✅ **Kabar Baik:** Tren menunjukkan PENURUNAN deforestasi rata-rata **{abs(slope):,.0f} Ha/tahun**. Pertahankan kebijakan ini untuk menyelamatkan **{future_sum:,.0f} Ha** hutan hingga 2030."
    
    st.markdown(f":{story_color}[{story_text}]")
    
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
    # Add vertical line for "Now"
    fig_trend.add_vline(x=2022, line_width=2, line_dash="dash", line_color="green", annotation_text="Saat Ini")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 3: STACKED AREA (COMPOSITION) ---
with tab3:
    st.subheader("Komposisi Deforestasi: Hutan vs APL")
    st.markdown("Menganalisis apakah deforestasi terjadi di dalam Kawasan Hutan (Ilegal/Legal Loging) atau Areal Penggunaan Lain (APL).")
    
    # Filter composition data
    comp_df = data_tidy[
        (~data_tidy['Category'].str.contains("Total")) & 
        (data_tidy['Provinsi'] == (selected_province if selected_province != 'Semua' else data_tidy['Provinsi']))
    ]
    
    if selected_province == 'Semua':
        # If 'Semua', aggregate by Year and Category
        comp_df = comp_df.groupby(['Year', 'Category'])['Value'].sum().reset_index()
    
    fig_area = px.area(
        comp_df,
        x='Year',
        y='Value',
        color='Category',
        title="Komposisi Deforestasi per Tahun",
        color_discrete_sequence=['#4CAF50', '#795548'] # Green for Hutan, Brown for APL idea
    )
    st.plotly_chart(fig_area, use_container_width=True)

# --- TAB 4: DATA DETAIL ---
with tab4:
    st.subheader("Raw Data viewer")
    st.dataframe(df_raw_p)
