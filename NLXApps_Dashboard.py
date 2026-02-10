import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- [1] 데이터 전처리 및 스마트 레이어 분석 로직 ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 1. 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 2. 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 3. 이상치 제거 (IQR)
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 4. 스마트 레이어 분석 (Z-Gap Detection)
    # 0, 175, 349 등 큰 간격을 찾아 자동으로 층 분리
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['Bump_Center_Z'].unique())
        z_diffs = np.diff(z_vals)
        # 50um 이상의 간격이 생기면 다른 층으로 인식
        gap_threshold = 50.0 
        split_points = z_vals[1:][z_diffs > gap_threshold]
        
        layer_assignment = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layer_assignment[df_clean['Bump_Center_Z'] >= p] += 1
        df_clean['Layer'] = layer_assignment
    else:
        df_clean['Layer'] = 1

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Expert", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Multi-Layer)")

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

# 그래프 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 12)
p_h = st.sidebar.slider("Plot Height", 3, 15, 6)
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['Layer'].unique())
        
        # 상단 탭 구성
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "🔄 Multi-Layer Shift"])

        # --- Tab 1: 단일 층 분석 ---
        with tab1:
            selected_layer = st.selectbox("Select Layer to View", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]
            
            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)
            fig, ax = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                sc = ax.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15)
                plt.colorbar(sc, label=f"{d_type} Value")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='Source', y='Value', ax=ax)
            elif chart_type == "Distribution":
                sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax)
            
            ax.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig)

        # --- Tab 2: 층별 비교 (Boxplot) ---
        with tab2:
            if len(unique_layers) > 1:
                st.subheader("Layer-wise Comparison")
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                # Layer를 X축으로 하여 비교
                sns.boxplot(data=combined_df, x='Layer', y='Value', hue='Source', ax=ax2)
                ax2.set_title("Value Comparison Across Layers")
                st.pyplot(fig2)
            else:
                st.info("데이터에 층이 하나만 존재하여 비교 그래프를 생성할 수 없습니다.")

        # --- Tab 3: Multi-Layer Shift 분석 ---
        with tab3:
            if len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Shift (Ref: Layer 1)")
                
                # 층간 Bump 매칭을 위해 좌표 라운딩 (미세 오차 허용)
                combined_df['X_id'] = combined_df['X'].round(1)
                combined_df['Y_id'] = combined_df['Y'].round(1)
                
                # Layer 1을 기준으로 나머지 층 비교
                base_layer = combined_df[combined_df['Layer'] == 1][['X_id', 'Y_id', 'X', 'Y', 'Source']]
                target_layer = combined_df[combined_df['Layer'] > 1]
                
                # 병합하여 차이 계산
                merged = pd.merge(base_layer, target_layer, on=['X_id', 'Y_id', 'Source'], suffixes=('_L1', '_LN'))
                merged['DX'] = merged['X_LN'] - merged['X_L1']
                merged['DY'] = merged['Y_LN'] - merged['Y_L1']
                merged['Alignment_Shift'] = np.sqrt(merged['DX']**2 + merged['DY']**2)
                
                # 시각화
                fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                sns.scatterplot(data=merged, x='DX', y='DY', hue='Layer', ax=ax3, alpha=0.7)
                ax3.axhline(0, color='black', linestyle='--')
                ax3.axvline(0, color='black', linestyle='--')
                ax3.set_title("Alignment Shift (Layer N vs Layer 1)")
                st.pyplot(fig3)
                
                st.write("**Shift Statistics (um)**")
                st.dataframe(merged.groupby(['Source', 'Layer'])['Alignment_Shift'].describe())
            else:
                st.info("Multi-layer 분석을 위해서는 2개 이상의 층이 필요합니다.")