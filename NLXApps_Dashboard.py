import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# --- [1] 데이터 전처리 및 자동 레이어 분석 로직 ---
def process_data(df, scale_factor, apply_iqr, apply_pitch_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Z_raw'] = df['Bump_Center_Z'] # 클러스터링용 원본 Z
    df['Value'] = df[target] * scale_factor
    
    # 1차 필터링: Value IQR
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # [자동 레이어 분석] Z값 차이가 확연하므로 클러스터링 최적화
    # 데이터 내 Z값의 고유한 분포를 분석하여 층수 자동 결정 (Gap 0.05mm 이상 기준)
    z_sorted = np.sort(df_clean['Z_raw'].unique())
    if len(z_sorted) > 1:
        z_diffs = np.diff(z_sorted)
        # 층 사이의 간격이 0.05 이상인 곳을 기준으로 층 개수 산정
        n_auto_layers = len([d for d in z_diffs if d > 0.05]) + 1
    else:
        n_auto_layers = 1

    # KMeans로 층 할당
    kmeans = KMeans(n_clusters=n_auto_layers, random_state=42, n_init=10)
    df_clean['Layer'] = kmeans.fit_predict(df_clean[['Z_raw']])
    
    # 낮은 Z값이 0층이 되도록 정렬
    layer_order = df_clean.groupby('Layer')['Z_raw'].mean().sort_values().index
    layer_map = {old: new for new, old in enumerate(layer_order)}
    df_clean['Layer'] = df_clean['Layer'].map(layer_map)

    # Pitch 계산 (X, Y)
    df_clean['Y_grid'] = df_clean['Y'].round(1)
    df_clean = df_clean.sort_values(by=['Y_grid', 'X'])
    df_clean['X_Pitch'] = df_clean.groupby('Y_grid')['X'].diff()

    df_clean['X_grid'] = df_clean['X'].round(1)
    df_clean = df_clean.sort_values(by=['X_grid', 'Y'])
    df_clean['Y_Pitch'] = df_clean.groupby('X_grid')['Y'].diff()

    if apply_pitch_iqr:
        for col in ['X_Pitch', 'Y_Pitch']:
            p_data = df_clean[col].dropna()
            if not p_data.empty:
                pq1, pq3 = p_data.quantile([0.25, 0.75])
                piqr = pq3 - pq1
                df_clean.loc[(df_clean[col] < pq1 - 1.5 * piqr) | (df_clean[col] > pq3 + 1.5 * piqr), col] = np.nan

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Auto-Layer Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis (Automatic Layer Detection)")

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Scale Factor (e.g., 1000 for mm to um)", value=1000)

use_val_iqr = st.sidebar.checkbox("Apply Value IQR", value=True)
use_pitch_iqr = st.sidebar.checkbox("Apply Pitch IQR", value=True)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_val_iqr, use_pitch_iqr)
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        
        # 레이어 선택 UI
        unique_layers = sorted(combined_df['Layer'].unique())
        selected_layer = st.sidebar.selectbox("Select Layer", ["All"] + [f"Layer {i}" for i in unique_layers])

        # 필터링
        display_df = combined_df if selected_layer == "All" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]

        # --- 요약 통계 ---
        st.subheader(f"📊 Statistics: {selected_layer}")
        col1, col2 = st.columns([1, 2])
        
        summary = display_df.groupby('Source').agg({
            'Value': ['mean', 'std', 'count'],
            'X_Pitch': 'mean',
            'Y_Pitch': 'mean'
        }).reset_index()
        st.dataframe(summary)

        # --- 그래프 출력 ---
        st.markdown("---")
        st.subheader("📈 Visualization")
        
        chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "Heatmap":
            # 2D 산점도로 히트맵 구현
            sc = ax.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=10)
            plt.colorbar(sc, label=f"{d_type} Value")
            ax.set_title(f"{d_type} Top View Map")
            
        elif chart_type == "Box Plot":
            sns.boxplot(data=display_df, x='Source', y='Value', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
        elif chart_type == "Distribution":
            sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax)
            
        st.pyplot(fig)