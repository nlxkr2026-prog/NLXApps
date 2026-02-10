import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 로직 (일반/Multi-layer 대응) ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명 표준화 (대문자)
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 2. 좌표 및 측정값 설정
    # X_COORD가 있으면 우선 사용, 없으면 BUMP_CENTER_X 사용
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    df['MEAS_VALUE'] = df[target] * scale_factor
    
    # 3. 레이어 번호 설정 (Pillar 데이터인 경우 LAYER_NUMBER 우선)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        if len(z_vals) > 1:
            z_diffs = np.diff(z_vals)
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            l_assign = np.ones(len(df), dtype=int)
            for p in splits: l_assign[df['BUMP_CENTER_Z'] >= p] += 1
            df['L_NUM'] = l_assign
        else:
            df['L_NUM'] = 0 # 일반 데이터는 0 혹은 1로 시작
    else:
        df['L_NUM'] = 0

    # 4. Pillar 식별자 (Shift 분석용)
    if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
    elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
    else: df['P_ID'] = None

    # 5. IQR 필터링 (측정값이 좌표가 아닐 때만 적용)
    df_clean = df.copy()
    if apply_iqr and d_type != "Coordinate":
        df_clean = df_clean[df_clean['MEAS_VALUE'] != 0]
        if not df_clean.empty:
            q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Bump Expert", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

st.sidebar.header("📁 Data Setting")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

# 그래프 크기 및 라벨 설정
p_w = st.sidebar.slider("Plot Width", 5, 25, 10)
p_h = st.sidebar.slider("Plot Height", 3, 15, 8)
custom_title = st.sidebar.text_input("Graph Title", "Alignment Analysis")

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
            plt.colorbar(sc, label=f"{d_type} Value")
            ax1.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                st.pyplot(fig2)
                st.dataframe(combined_df.groupby(['SOURCE_FILE', 'L_NUM'])['MEAS_VALUE'].agg(['mean', 'std', 'count']))
            else:
                st.info("비교를 위해 2개 이상의 레이어가 필요합니다.")

        # --- Tab 3: Multi-Layer Shift (로직 충돌 방지 적용) ---
        with tab3:
            # P_ID(Pillar)가 있는 데이터만 분석 수행
            shift_capable_df = combined_df.dropna(subset=['P_ID'])
            
            if not shift_capable_df.empty and len(unique_layers) > 1:
                st.subheader("Pillar-wise Shift Trend (Ref: Layer 0)")
                trend_list = []
                for src in shift_capable_df['SOURCE_FILE'].unique():
                    src_df = shift_capable_df[shift_capable_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    
                    if not base.empty:
                        for lyr in unique_layers:
                            target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                            merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                            if not merged.empty:
                                dx = (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean()
                                dy = (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()
                                trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': dx, 'Avg_DY': dy})
                
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y)")
                    ax3.axvline(0, color='black', alpha=0.3)
                    ax3.set_yticks(unique_layers)
                    ax3.set_xlabel("Average Shift (um)"); ax3.set_ylabel("Layer")
                    ax3.legend()
                    st.pyplot(fig3)
                else:
                    st.warning("Layer 0를 포함한 Pillar 데이터가 부족합니다.")
            else:
                st.info("Pillar 기반 Shift 분석을 위해 Coordinate 데이터가 필요합니다.")