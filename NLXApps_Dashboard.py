import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- [1] 데이터 통합 처리 엔진 ---
def load_and_detect_mode(files, scale_factor, layer_gap):
    all_processed = []
    global_mode = "Single" # 기본값
    
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # A. 모드 판별
        if 'PILLAR_NUMBER' in df.columns and 'LAYER_NUMBER' in df.columns:
            mode = "Multi-Shift"
            global_mode = "Multi-Shift"
        else:
            mode = "Single"
        
        # B. 좌표 표준화 (X_VAL, Y_VAL, Z_VAL)
        # Multi-Shift용 좌표
        if 'X_COORD' in df.columns: df['X_VAL'] = df['X_COORD'] * scale_factor
        elif 'BUMP_CENTER_X' in df.columns: df['X_VAL'] = df['BUMP_CENTER_X'] * scale_factor
        
        if 'Y_COORD' in df.columns: df['Y_VAL'] = df['Y_COORD'] * scale_factor
        elif 'BUMP_CENTER_Y' in df.columns: df['Y_VAL'] = df['BUMP_CENTER_Y'] * scale_factor
        
        if 'Z_COORD' in df.columns: df['Z_VAL'] = df['Z_COORD'] * scale_factor
        elif 'BUMP_CENTER_Z' in df.columns: df['Z_VAL'] = df['BUMP_CENTER_Z'] * scale_factor
        else: df['Z_VAL'] = 0
        
        # C. 레이어 결정
        if 'LAYER_NUMBER' in df.columns:
            df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
        else:
            # Z값 기반 자동 레이어링
            z_arr = np.sort(df['Z_VAL'].unique())
            if len(z_arr) > 1:
                diffs = np.diff(z_arr)
                splits = z_arr[1:][diffs > layer_gap]
                l_assign = np.zeros(len(df), dtype=int)
                for s in splits: l_assign[df['Z_VAL'] >= s] += 1
                df['L_NUM'] = l_assign
            else:
                df['L_NUM'] = 0
        
        # D. 식별자 고정
        df['ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else df.get('GROUP_ID', df.index)
        df['SOURCE'] = os.path.splitext(f.name)[0]
        all_processed.append(df)
        
    return pd.concat(all_processed, ignore_index=True) if all_processed else None, global_mode

# --- [2] UI & 분석 로직 ---
st.set_page_config(page_title="NLX Unified Dashboard", layout="wide")
st.title("🔬 NLX Unified Bump Analysis")

with st.sidebar:
    st.header("⚙️ Global Settings")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
    multiplier = st.number_input("Unit Multiplier (e.g., mm to um = 1000)", value=1, step=1)
    gap = st.number_input("Layer Detection Gap (um)", value=5.0, step=0.5)
    
    if uploaded_files:
        raw_combined, data_mode = load_and_detect_mode(uploaded_files, multiplier, gap)
        st.info(f"Detected Mode: **{data_mode}**")
        
        # 분석 대상 컬럼 선택 (Pivot 기능)
        exclude = ['X_VAL', 'Y_VAL', 'Z_VAL', 'L_NUM', 'ID', 'SOURCE', 'GROUP_ID', 'PILLAR_NUMBER', 'LAYER_NUMBER', 'X_COORD', 'Y_COORD', 'Z_COORD']
        available_cols = [c for c in raw_combined.columns if c not in exclude and not c.endswith('_VAL')]
        target_col = st.selectbox("Select Target Measurement", available_cols)
        
        st.markdown("---")
        p_w = st.slider("Plot Width", 5, 25, 12)
        p_h = st.slider("Plot Height", 3, 15, 6)

if uploaded_files and raw_combined is not None:
    # 데이터 준비
    df = raw_combined.copy()
    # 선택된 타겟 데이터 스케일링 (이미 좌표는 처리됨)
    df['VALUE'] = df[target_col] * multiplier
    
    # 탭 구성
    if data_mode == "Single":
        tabs = st.tabs(["📊 Distribution", "🗺️ Heatmap", "📈 Comparison", "🎯 Pitch", "🧊 3D View"])
    else:
        tabs = st.tabs(["📊 Distribution", "📉 Shift Trend", "🧊 3D View"])

    # --- 공통 탭: Distribution (Histogram & Box) ---
    with tabs[0]:
        sel_layer = st.selectbox("Select Layer", ["All"] + [f"L{i}" for i in sorted(df['L_NUM'].unique())])
        plot_df = df if sel_layer == "All" else df[df['L_NUM'] == int(sel_layer[1:])]
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(p_w/2, p_h))
            sns.histplot(data=plot_df, x='VALUE', hue='SOURCE', kde=True, ax=ax)
            ax.set_title(f"{target_col} Distribution")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(p_w/2, p_h))
            sns.boxplot(data=plot_df, x='L_NUM', y='VALUE', hue='SOURCE', ax=ax)
            ax.set_title(f"{target_col} by Layer")
            st.pyplot(fig)

    # --- Single 전용 탭: Heatmap ---
    if data_mode == "Single":
        with tabs[1]:
            st.subheader(f"{target_col} Spatial Heatmap")
            fig = px.scatter(df, x='X_VAL', y='Y_VAL', color='VALUE', facet_col='L_NUM', 
                             hover_data=['ID'], color_continuous_scale='viridis')
            fig.update_layout(height=500); st.plotly_chart(fig, use_container_width=True)

        with tabs[3]: # Pitch Analysis
            st.subheader("🎯 Pitch Analysis (Logic: Sorted Grid Diff)")
            # 간단 Pitch 로직: Row/Col별 차이
            df_p = df.sort_values(['L_NUM', 'SOURCE', 'Y_VAL', 'X_VAL'])
            df_p['X_PITCH'] = df_p.groupby(['L_NUM', 'SOURCE', 'Y_VAL'])['X_VAL'].diff().abs()
            
            fig, ax = plt.subplots(figsize=(p_w, p_h))
            sns.boxplot(data=df_p, x='SOURCE', y='X_PITCH', hue='L_NUM', ax=ax)
            st.pyplot(fig)

    # --- Multi-Shift 전용 탭: Shift Trend ---
    else:
        with tabs[1]:
            st.subheader("📉 Layer-to-Layer Shift Trend (Relative to L0)")
            trend_data = []
            for src in df['SOURCE'].unique():
                src_df = df[df['SOURCE'] == src]
                base = src_df[src_df['L_NUM'] == 0][['ID', 'X_VAL', 'Y_VAL']]
                for lyr in sorted(df['L_NUM'].unique()):
                    if lyr == 0: continue
                    target = src_df[src_df['L_NUM'] == lyr][['ID', 'X_VAL', 'Y_VAL']]
                    m = pd.merge(base, target, on='ID', suffixes=('_0', '_L'))
                    if not m.empty:
                        trend_data.append({'Source': src, 'Layer': lyr, 
                                           'DX': (m['X_VAL_L'] - m['X_VAL_0']).mean(),
                                           'DY': (m['Y_VAL_L'] - m['Y_VAL_0']).mean()})
            if trend_data:
                tdf = pd.DataFrame(trend_data)
                fig = px.line(tdf, x='DX', y='Layer', color='Source', markers=True, title="X-Shift Trend")
                st.plotly_chart(fig, use_container_width=True)

    # --- 공통 탭: 3D View ---
    with tabs[-1]:
        st.subheader("🧊 3D Stack View")
        fig = px.scatter_3d(df, x='X_VAL', y='Y_VAL', z='Z_VAL', color='VALUE', 
                            opacity=0.7, color_continuous_scale='Turbo')
        fig.update_layout(height=700); st.plotly_chart(fig, use_container_width=True)

else:
    st.info("파일을 업로드하면 자동으로 모드를 분석하여 대시보드를 생성합니다.")