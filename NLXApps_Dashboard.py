import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# --- [1] 데이터 전처리 로직 (Shift 데이터 대응 및 단일층 예외처리) ---
def process_data(df, scale_factor, apply_iqr, apply_pitch_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 1. 데이터 타입 및 타겟 설정
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 2. 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 3. 이상치 제거 (Value)
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 4. 레이어 분석 (Z 데이터가 있을 때만 수행)
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_values = df_clean['Bump_Center_Z'].unique()
        z_sorted = np.sort(z_values)
        z_diffs = np.diff(z_sorted)
        n_auto_layers = len([d for d in z_diffs if d > 0.05]) + 1
        
        kmeans = KMeans(n_clusters=n_auto_layers, random_state=42, n_init=10)
        df_clean['Layer'] = kmeans.fit_predict(df_clean[['Bump_Center_Z']])
        
        # 낮은 층부터 0, 1, 2... 순서 부여
        layer_order = df_clean.groupby('Layer')['Bump_Center_Z'].mean().sort_values().index
        df_clean['Layer'] = df_clean['Layer'].map({old: new for new, old in enumerate(layer_order)})
    else:
        # Z 데이터가 없거나 값이 하나뿐이면 단일 층으로 간주
        df_clean['Layer'] = 0

    # 5. Pitch 계산
    df_clean['Y_grid'] = df_clean['Y'].round(1)
    df_clean = df_clean.sort_values(by=['Y_grid', 'X'])
    df_clean['X_Pitch'] = df_clean.groupby('Y_grid')['X'].diff()

    df_clean['X_grid'] = df_clean['X'].round(1)
    df_clean = df_clean.sort_values(by=['X_grid', 'Y'])
    df_clean['Y_Pitch'] = df_clean.groupby('X_grid')['Y'].diff()

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Advanced Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

# 사이드바 설정
st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)

# [추가] 그래프 커스터마이징 기능
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_x_label = st.sidebar.text_input("X-axis Label", "X Position (um)")
custom_y_label = st.sidebar.text_input("Y-axis Label", "Y Position (um)")

col_min, col_max = st.sidebar.columns(2)
use_custom_scale = st.sidebar.checkbox("Apply Custom Value Scale")
v_min = col_min.number_input("Value Min", value=0.0)
v_max = col_max.number_input("Value Max", value=20.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, True, True)
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['Layer'].unique())
        selected_layer = st.sidebar.selectbox("Select Layer", ["All"] + [f"Layer {i}" for i in unique_layers])

        # 레이어 필터링
        display_df = combined_df if selected_layer == "All" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]

        # --- 메인 그래프 영역 ---
        st.subheader(f"📊 {d_type} Analysis Results ({selected_layer})")
        
        chart_type = st.radio("Select View", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "Heatmap":
            # 색상 범위 설정
            vmin_val = v_min if use_custom_scale else display_df['Value'].min()
            vmax_val = v_max if use_custom_scale else display_df['Value'].max()
            
            sc = ax.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], 
                            cmap='jet', s=15, vmin=vmin_val, vmax=vmax_val)
            plt.colorbar(sc, label=f"{d_type} ({'um' if scale==1000 else 'unit'})")
            ax.set_xlabel(custom_x_label)
            ax.set_ylabel(custom_y_label)
            ax.set_title(f"{d_type} Map")

        elif chart_type == "Box Plot":
            sns.boxplot(data=display_df, x='Source', y='Value', ax=ax)
            ax.set_ylabel(f"{d_type} Value")
            if use_custom_scale: ax.set_ylim(v_min, v_max)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        elif chart_type == "Distribution":
            sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax)
            ax.set_xlabel(f"{d_type} Value")
            if use_custom_scale: ax.set_xlim(v_min, v_max)

        st.pyplot(fig)

        # 요약 통계 테이블
        st.markdown("---")
        st.subheader("📋 Summary Statistics")
        summary_df = display_df.groupby('Source')['Value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        st.dataframe(summary_df, use_container_width=True)