import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- [1] 데이터 전처리 엔진 ---
def process_data(files, multiplier, layer_gap, apply_iqr, pitch_sens):
    all_dfs = []
    
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 1. 좌표 및 기본 컬럼 표준화
        # X, Y 좌표 (mm -> um 등 Multiplier 적용)
        if 'X_COORD' in df.columns: df['X_VAL'] = df['X_COORD'] * multiplier
        elif 'BUMP_CENTER_X' in df.columns: df['X_VAL'] = df['BUMP_CENTER_X'] * multiplier
        
        if 'Y_COORD' in df.columns: df['Y_VAL'] = df['Y_COORD'] * multiplier
        elif 'BUMP_CENTER_Y' in df.columns: df['Y_VAL'] = df['BUMP_CENTER_Y'] * multiplier
        
        if 'Z_COORD' in df.columns: df['Z_VAL'] = df['Z_COORD'] * multiplier
        elif 'BUMP_CENTER_Z' in df.columns: df['Z_VAL'] = df['BUMP_CENTER_Z'] * multiplier
        else: df['Z_VAL'] = 0
        
        # 2. 레이어 구분 (명시적 컬럼 우선, 없으면 Z값 기반 자동)
        if 'LAYER_NUMBER' in df.columns:
            df['L_NUM'] = df['LAYER_NUMBER'].fillna(0).astype(int)
        else:
            z_arr = np.sort(df['Z_VAL'].unique())
            if len(z_arr) > 1:
                diffs = np.diff(z_arr)
                splits = z_arr[1:][diffs > layer_gap]
                l_assign = np.zeros(len(df), dtype=int)
                for s in splits: l_assign[df['Z_VAL'] >= s] += 1
                df['L_NUM'] = l_assign
            else:
                df['L_NUM'] = 0
                
        # 3. 고유 식별자 설정 (Pillar > Group > Index)
        if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
        elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
        else: df['P_ID'] = df.index
        
        df['SOURCE_FILE'] = os.path.splitext(f.name)[0]
        all_dfs.append(df)
        
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# --- [2] Pitch 계산 로직 (Missing Bump 고려) ---
def calculate_pitch(df, pitch_sens):
    grid_size = 0.5 # 정렬 오차 허용 그리드
    group_cols = ['SOURCE_FILE', 'L_NUM']
    
    # X-Pitch (Y 그리드 내 정렬)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_cols + ['Y_GRID', 'X_VAL'])
    # ID가 연속적일 때만 유효 Pitch로 간주
    df['ID_DIFF'] = df.groupby(group_cols + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = np.where(df['ID_DIFF'] == 1, df.groupby(group_cols + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)
    
    # Y-Pitch (X 그리드 내 정렬)
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(group_cols + ['X_GRID'])['Y_VAL'].diff().abs()
    
    # 통계적 필터링 (배수 Pitch 제거 및 IQR)
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg = v.mean()
            # 평균 1.5배 이상(Missing) 혹은 0.5배 미만(Noise) 제거
            df[col] = np.where((df[col] > avg*1.5) | (df[col] < avg*0.5), np.nan, df[col])
            # IQR 필터 적용
            q1, q3 = v.quantile([0.25, 0.75])
            df.loc[(df[col] < q1 - pitch_sens*(q3-q1)) | (df[col] > q3 + pitch_sens*(q3-q1)), col] = np.nan
    return df

# --- [3] UI & 메인 로직 ---
st.set_page_config(page_title="NLX Unified Analysis", layout="wide")
st.title("🔬 NLX Unified Bump Analysis Dashboard")

with st.sidebar:
    st.header("📁 Data Config")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
    multiplier = st.number_input("Unit Multiplier (Int)", value=1, step=1, format="%d")
    layer_gap = st.number_input("Layer Detection Gap (um)", value=5.0, step=0.5)
    
    if uploaded_files:
        # 데이터 1차 로드
        combined_df = process_data(uploaded_files, multiplier, layer_gap, True, 1.2)
        
        # 분석 대상 컬럼 선택 (Pivot)
        exclude_cols = ['X_VAL', 'Y_VAL', 'Z_VAL', 'L_NUM', 'P_ID', 'SOURCE_FILE', 'Y_GRID', 'X_GRID', 'ID_DIFF']
        available_targets = [c for c in combined_df.columns if c not in exclude_cols and not c.endswith('_UM')]
        target_col = st.selectbox("Select Target Measurement", available_targets)
        
        st.markdown("---")
        st.subheader("🧹 Filters")
        use_iqr = st.checkbox("Apply IQR Filter to Target", value=True)
        pitch_sens = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2)
        
        st.markdown("---")
        st.subheader("📏 Scale & Legend")
        show_legend = st.checkbox("Show Legend", value=True)
        leg_loc = st.selectbox("Legend Loc", ["best", "upper right", "right", "center"], index=1)
        use_manual_axis = st.checkbox("Manual Axis Range", value=False)
        v_min = st.number_input("Min", value=-10.0)
        v_max = st.number_input("Max", value=10.0)

if uploaded_files and combined_df is not None:
    # 1. 데이터 정제 (선택된 타겟 기준)
    df = combined_df.copy()
    df['TARGET_VAL'] = df[target_col] * multiplier
    
    if use_iqr:
        q1, q3 = df['TARGET_VAL'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df['TARGET_VAL'] >= q1 - 1.5*iqr) & (df['TARGET_VAL'] <= q3 + 1.5*iqr)]
    
    # 2. Pitch 계산
    df = calculate_pitch(df, pitch_sens)
    
    # 3. 모드 판별 (Multi-Layer Shift 여부)
    is_multi_shift = 'LAYER_NUMBER' in df.columns and 'PILLAR_NUMBER' in df.columns
    
    # 탭 구성
    tab_list = ["📊 Statistics", "🗺️ Heatmap", "🎯 Pitch Analysis", "🧊 3D View"]
    if is_multi_shift: tab_list.insert(2, "📉 Shift Trend")
    tabs = st.tabs(tab_list)

    # --- Tab 0: Statistics (Distributions) ---
    with tabs[0]:
        sel_layer = st.selectbox("Select Layer View", ["All Layers"] + [f"Layer {i}" for i in sorted(df['L_NUM'].unique())])
        plot_df = df if sel_layer == "All Layers" else df[df['L_NUM'] == int(sel_layer.split(" ")[1])]
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=plot_df, x='TARGET_VAL', hue='SOURCE_FILE', kde=True, ax=ax)
            if use_manual_axis: ax.set_xlim(v_min, v_max)
            if show_legend: ax.legend(loc=leg_loc, title=None)
            st.pyplot(fig)
        with col2:
            st.markdown(f"**{target_col} Summary Stats (um)**")
            stats = plot_df.groupby(['SOURCE_FILE', 'L_NUM'])['TARGET_VAL'].agg(['mean', 'std', 'min', 'max', 'count'])
            st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)

    # --- Tab 1: Heatmap ---
    with tabs[1]:
        fig = px.scatter(df, x='X_VAL', y='Y_VAL', color='TARGET_VAL', facet_col='L_NUM',
                         hover_data=['P_ID'], color_continuous_scale='viridis', 
                         title=f"{target_col} Spatial Distribution")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Shift Trend (Only for Multi-Layer Shift Mode) ---
    if is_multi_shift:
        with tabs[2]:
            st.subheader("📉 Layer-to-Layer Shift Trend")
            trend_list = []
            for src in df['SOURCE_FILE'].unique():
                src_df = df[df['SOURCE_FILE'] == src]
                base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                for lyr in sorted(df['L_NUM'].unique()):
                    if lyr == 0: continue
                    target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                    merged = pd.merge(base, target, on='P_ID', suffixes=('_0', '_L'))
                    if not merged.empty:
                        trend_list.append({'Source': src, 'Layer': lyr, 
                                           'DX': (merged['X_VAL_L'] - merged['X_VAL_0']).mean(),
                                           'DY': (merged['Y_VAL_L'] - merged['Y_VAL_0']).mean()})
            if trend_list:
                tdf = pd.DataFrame(trend_list)
                c1, c2 = st.columns(2)
                with c1:
                    fig_x = px.line(tdf, x='DX', y='Layer', color='Source', markers=True, title="Average X-Shift by Layer")
                    if use_manual_axis: fig_x.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_x, use_container_width=True)
                with c2:
                    fig_y = px.line(tdf, x='DY', y='Layer', color='Source', markers=True, title="Average Y-Shift by Layer")
                    if use_manual_axis: fig_y.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_y, use_container_width=True)

    # --- Tab: Pitch Analysis ---
    pitch_tab_idx = 3 if is_multi_shift else 2
    with tabs[pitch_tab_idx]:
        st.subheader("🎯 Pitch Analysis (X & Y)")
        col_px, col_py = st.columns(2)
        with col_px:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='SOURCE_FILE', y='X_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("X-Pitch Distribution")
            st.pyplot(fig)
        with col_py:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='SOURCE_FILE', y='Y_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("Y-Pitch Distribution")
            st.pyplot(fig)

    # --- Tab: 3D View ---
    with tabs[-1]:
        st.subheader("🧊 3D Stacked View")
        fig_3d = px.scatter_3d(df, x='X_VAL', y='Y_VAL', z='Z_VAL', color='TARGET_VAL',
                               opacity=0.7, color_continuous_scale='Turbo', hover_data=['P_ID', 'L_NUM'])
        fig_3d.update_layout(height=800)
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("Please upload CSV files to start analysis.")