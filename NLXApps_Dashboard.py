import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 로직 (사용자 컬럼명 최우선 대응) ---
def process_data(df, scale_factor, apply_iqr):
    # 모든 컬럼명을 대문자로 변환하여 오타 및 대소문자 문제 해결
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별 및 타겟 설정
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 2. 좌표 데이터 설정 (X_COORD, Y_COORD 등)
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df['BUMP_CENTER_X']) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df['BUMP_CENTER_Y']) * scale_factor
    df['MEAS_VALUE'] = df[target] * scale_factor
    
    # 3. 레이어 번호 매핑 (LAYER_NUMBER 우선)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'LAYER' in df.columns:
        df['L_NUM'] = df['LAYER'].astype(int)
    else:
        # Z 좌표 기반 자동 레이어링 (Z_COORD 또는 BUMP_CENTER_Z)
        z_col = 'Z_COORD' if 'Z_COORD' in df.columns else ('BUMP_CENTER_Z' if 'BUMP_CENTER_Z' in df.columns else None)
        if z_col:
            z_vals = np.sort(df[z_col].unique())
            z_diffs = np.diff(z_vals)
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            l_assign = np.ones(len(df), dtype=int)
            for p in splits: l_assign[df[z_col] >= p] += 1
            df['L_NUM'] = l_assign
        else:
            df['L_NUM'] = 1

    # 4. Pillar 식별자 매핑 (PILLAR_NUMBER 우선)
    if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
    elif 'PILLAR' in df.columns: df['P_ID'] = df['PILLAR']
    elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
    else: df['P_ID'] = df.index

    # 5. 데이터 필터링 (Coordinate 타입일 경우 0을 삭제하지 않음)
    if d_type == "Coordinate":
        df_clean = df.copy()
    else:
        df_clean = df[df['MEAS_VALUE'] != 0].copy()
    
    if apply_iqr and d_type != "Coordinate":
        q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Expert Dashboard", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

# 사이드바 설정
st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 10)
p_h = st.sidebar.slider("Plot Height", 3, 15, 8)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")
custom_x_label = st.sidebar.text_input("X-axis Label", "Shift Value (um)")
custom_y_label = st.sidebar.text_input("Y-axis Label", "Layer Number")

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE_FILE'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())
        
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Shift Trend"])

        with tab1:
            selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(selected_layer.split(" ")[1])]
            
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sc = ax1.scatter(display_df['X_VAL'], display_df['Y_VAL'], c=display_df['MEAS_VALUE'], cmap='jet', s=15)
            plt.colorbar(sc, label=f"{d_type}")
            ax1.set_title(f"{custom_title} ({selected_layer})")
            ax1.set_xlabel("X (um)"); ax1.set_ylabel("Y (um)")
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                ax2.set_xlabel("Layer Number"); ax2.set_ylabel(f"{d_type} Value")
                ax2.set_title(f"Layer Comparison: {custom_title}")
                st.pyplot(fig2)

        with tab3:
            if len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Shift (Y: Layer, X: Avg Shift)")
                trend_list = []
                for src in combined_df['SOURCE_FILE'].unique():
                    src_df = combined_df[combined_df['SOURCE_FILE'] == src]
                    # Layer 1을 기준(Reference)으로 설정
                    base = src_df[src_df['L_NUM'] == 1][['P_ID', 'X_VAL', 'Y_VAL']]
                    for lyr in unique_layers:
                        target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                        merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                        if not merged.empty:
                            merged['DX'] = merged['X_VAL_TGT'] - merged['X_VAL_REF']
                            merged['DY'] = merged['Y_VAL_TGT'] - merged['Y_VAL_REF']
                            trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': merged['DX'].mean(), 'Avg_DY': merged['DY'].mean()})
                
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X Avg)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Avg)")
                    ax3.axvline(0, color='black', alpha=0.3)
                    ax3.set_yticks(unique_layers)
                    ax3.set_xlabel(custom_x_label); ax3.set_ylabel(custom_y_label)
                    ax3.set_title(f"{custom_title}: Vertical Shift Trend")
                    ax3.legend()
                    st.pyplot(fig3)
                    st.dataframe(trend_df)
                    st.download_button("📥 Download Trend CSV", trend_df.to_csv(index=False).encode('utf-8'), "shift_trend.csv")
            else:
                st.info("Shift 트렌드 분석을 위해서는 2층 이상의 데이터가 필요합니다.")