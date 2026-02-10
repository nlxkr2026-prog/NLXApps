import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def process_data(df, scale_factor, apply_iqr):
    # 컬럼명 대문자 표준화 및 공백 제거
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 분석 대상 컬럼(Value) 찾기
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 2. X, Y 좌표 표준화
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df['BUMP_CENTER_X']) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df['BUMP_CENTER_Y']) * scale_factor
    df['MEAS_VALUE'] = df[target] * scale_factor
    
    # 3. 레이어 번호 및 Pillar ID 추출
    df['L_NUM'] = (df['LAYER_NUMBER'] if 'LAYER_NUMBER' in df.columns else (df['LAYER'] if 'LAYER' in df.columns else 1)).astype(int)
    df['P_ID'] = df['PILLAR'] if 'PILLAR' in df.columns else (df['GROUP_ID'] if 'GROUP_ID' in df.columns else df.index)

    # 4. IQR 필터링
    df_clean = df[df['MEAS_VALUE'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- UI 설정 ---
st.set_page_config(page_title="NLX Multi-Layer Trend", layout="wide")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

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
        tab1, tab2, tab3 = st.tabs(["📊 Data View", "📈 Statistics", "📉 Multi-Layer Shift Trend"])

        with tab3:
            if len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Shift Trend (Relative to Layer 1)")
                trend_list = []
                for src in combined_df['SOURCE_FILE'].unique():
                    src_df = combined_df[combined_df['SOURCE_FILE'] == src]
                    # 기준이 되는 1층 데이터 추출
                    base = src_df[src_df['L_NUM'] == 1][['P_ID', 'X_VAL', 'Y_VAL']]
                    
                    for lyr in unique_layers:
                        # 해당 층의 데이터와 Pillar ID 기준으로 병합 (D3-D2 계산 준비)
                        target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                        merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                        
                        if not merged.empty:
                            merged['DX'] = merged['X_VAL_TGT'] - merged['X_VAL_REF']
                            merged['DY'] = merged['Y_VAL_TGT'] - merged['Y_VAL_REF']
                            trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': merged['DX'].mean(), 'Avg_DY': merged['DY'].mean()})
                
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X Shift)")
                        ax.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Shift)")
                    
                    ax.axvline(0, color='black', alpha=0.3)
                    ax.set_yticks(unique_layers)
                    ax.set_xlabel("Average Shift Value (um)")
                    ax.set_ylabel("Layer Number")
                    ax.legend()
                    st.pyplot(fig)