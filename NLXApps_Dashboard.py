import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 로직 (Z-Gap 0번부터 할당 및 대소문자 표준화) ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명 대문자 표준화
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별 및 타겟 설정
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 2. 좌표 및 측정값 설정
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    df['MEAS_VALUE'] = df[target] * scale_factor
    
    # 3. 레이어 번호 설정 (Layer 0부터 시작하도록 정렬)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        if len(z_vals) > 1:
            z_diffs = np.diff(z_vals)
            # 50um(0.05mm) 이상 차이날 경우 층 분리
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            
            # 0번 레이어부터 할당 시작
            l_assign = np.zeros(len(df), dtype=int)
            for p in splits:
                l_assign[df['BUMP_CENTER_Z'] >= p] += 1
            df['L_NUM'] = l_assign
        else:
            df['L_NUM'] = 0
    else:
        df['L_NUM'] = 0

    # 4. Pillar 식별자 (Shift 분석용)
    df['P_ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else (df['GROUP_ID'] if 'GROUP_ID' in df.columns else None)

    # 5. IQR 필터링 (측정값이 좌표인 경우 제외)
    df_clean = df.copy()
    if apply_iqr and d_type != "Coordinate":
        df_clean = df_clean[df_clean['MEAS_VALUE'] != 0]
        if not df_clean.empty:
            q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Professional", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Ref: Layer 0)")

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 10)
p_h = st.sidebar.slider("Plot Height", 3, 15, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Alignment Analysis")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale Range")
v_min = st.sidebar.number_input("Min Limit", value=-10.0)
v_max = st.sidebar.number_input("Max Limit", value=10.0)

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
            
            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Histogram"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X_VAL'], display_df['Y_VAL'], c=display_df['MEAS_VALUE'], cmap='jet', s=15)
                if use_custom_scale: sc.set_clim(v_min, v_max)
                plt.colorbar(sc, label=f"{d_type} Value")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='SOURCE_FILE', y='MEAS_VALUE', ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            elif chart_type == "Histogram":
                sns.histplot(data=display_df, x='MEAS_VALUE', hue='SOURCE_FILE', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
            
            ax1.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                ax2.set_title(f"Layer Comparison: {custom_title}")
                st.pyplot(fig2)
            else:
                st.info("비교를 위해서는 2층 이상의 레이어가 필요합니다.")

        with tab3:
            # Pillar 정보가 있는 데이터에 대해 Shift Trend 분석
            shift_df = combined_df.dropna(subset=['P_ID'])
            if not shift_df.empty and len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Trend (Reference: Layer 0)")
                trend_list = []
                for src in shift_df['SOURCE_FILE'].unique():
                    src_df = shift_df[shift_df['SOURCE_FILE'] == src]
                    # 기준이 되는 Layer 0 데이터 추출
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    
                    if not base.empty:
                        for lyr in unique_layers:
                            target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                            merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                            if not merged.empty:
                                # 1:1 매칭된 개별 Bump의 변화량 평균 계산
                                dx = (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean()
                                dy = (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()
                                trend_list.append({'Source': src, 'Layer': lyr, 'DX': dx, 'DY': dy})
                
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['DX'], data['Layer'], marker='o', label=f"{src} (X Avg)")
                        ax3.plot(data['DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Avg)")
                    
                    ax3.axvline(0, color='black', alpha=0.3)
                    ax3.set_yticks(unique_layers)
                    ax3.set_xlabel("Average Shift (um)"); ax3.set_ylabel("Layer Number")
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    ax3.legend()
                    st.pyplot(fig3)
                    st.download_button("📥 Export Trend CSV", trend_df.to_csv(index=False).encode('utf-8'), "alignment_trend.csv")
            else:
                st.info("Pillar 기반 Shift 분석을 위해 Coordinate 데이터가 필요하며, Layer 0가 포함되어야 합니다.")