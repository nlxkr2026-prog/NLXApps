import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- [1] 데이터 전처리 로직 (대소문자 무시 및 컬럼 표준화) ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명을 대문자로 통일하여 대소문자 구분 문제 원천 차단
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별 및 타겟 설정
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 2. 좌표 및 값 설정 (X_COORD, BUMP_CENTER_X 등 유연하게 대응)
    df['X'] = (df['X_COORD'] if 'X_COORD' in df.columns else df['BUMP_CENTER_X']) * scale_factor
    df['Y'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df['BUMP_CENTER_Y']) * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 3. 레이어 번호 표준화 (LAYER_NUMBER 또는 LAYER 또는 Z-Gap)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'LAYER' in df.columns:
        df['L_NUM'] = df['LAYER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        # Z축 기반 자동 레이어링 (이전 로직 유지)
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        z_diffs = np.diff(z_vals)
        gap = max((z_vals.max() - z_vals.min()) * 0.1, 50.0)
        splits = z_vals[1:][z_diffs > gap]
        l_assign = np.ones(len(df), dtype=int)
        for p in splits: l_assign[df['BUMP_CENTER_Z'] >= p] += 1
        df['L_NUM'] = l_assign
    else:
        df['L_NUM'] = 1

    # 4. IQR 필터링
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Expert", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

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
custom_x_label = st.sidebar.text_input("X-axis Legend", "Average Shift (um)")
custom_y_label = st.sidebar.text_input("Y-axis Legend", "Layer Number")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale")
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
            
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sc = ax1.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15)
            plt.colorbar(sc, label=f"{d_type} Value")
            ax1.set_title(f"{custom_title} ({selected_layer})")
            ax1.set_xlabel("X (um)"); ax1.set_ylabel("Y (um)")
            if use_custom_scale: sc.set_clim(v_min, v_max)
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='Value', hue='SOURCE_FILE', ax=ax2)
                ax2.set_xlabel("Layer Number")
                ax2.set_title(f"Comparison across Layers: {custom_title}")
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                st.pyplot(fig2)

        # --- Tab 3: 사용자 정의 Multi-Layer Shift 로직 (Pillar 기반 계산) ---
        with tab3:
            st.subheader("Pillar-based Multi-Layer Alignment Shift")
            
            # Pillar 식별 컬럼 찾기 (PILLAR 또는 GROUP_ID)
            p_col = 'PILLAR' if 'PILLAR' in combined_df.columns else ('GROUP_ID' if 'GROUP_ID' in combined_df.columns else None)
            
            if p_col and len(unique_layers) > 1:
                trend_results = []
                for src in combined_df['SOURCE_FILE'].unique():
                    src_df = combined_df[combined_df['SOURCE_FILE'] == src]
                    
                    # Layer 1 좌표를 기준(Base)으로 설정
                    base_coords = src_df[src_df['L_NUM'] == 1][[p_col, 'X', 'Y']]
                    
                    for lyr in unique_layers:
                        target_coords = src_df[src_df['L_NUM'] == lyr][[p_col, 'X', 'Y']]
                        # Pillar 번호를 기준으로 1:1 매칭 (Merge)
                        merged = pd.merge(base_coords, target_coords, on=p_col, suffixes=('_REF', '_TGT'))
                        
                        if not merged.empty:
                            # 개별 Pillar의 Shift 계산 (TGT - REF)
                            merged['DX'] = merged['X_TGT'] - merged['X_REF']
                            merged['DY'] = merged['Y_TGT'] - merged['Y_REF']
                            
                            # Pillar 전체 평균값 산출
                            avg_dx = merged['DX'].mean()
                            avg_dy = merged['DY'].mean()
                            trend_results.append({'Source': src, 'Layer': lyr, 'Avg_DX': avg_dx, 'Avg_DY': avg_dy})
                
                if trend_results:
                    trend_df = pd.DataFrame(trend_results)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X Avg)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Avg)")
                    
                    ax3.axvline(0, color='black', alpha=0.3)
                    ax3.set_yticks(unique_layers)
                    ax3.set_title(f"{custom_title}: Vertical Shift Trend")
                    ax3.set_xlabel(custom_x_label); ax3.set_ylabel(custom_y_label)
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    ax3.legend()
                    st.pyplot(fig3)
                    st.dataframe(trend_df)
            else:
                st.warning("분석을 위해 PILLAR(또는 GROUP_ID) 정보와 2개 이상의 레이어가 필요합니다.")