import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- [1] 데이터 전처리 엔진 ---
def process_data(files, multiplier, layer_gap):
    all_dfs = []
    
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 1. 좌표 표준화 (Multiplier 적용)
        # X Coord
        if 'X_COORD' in df.columns: df['X_VAL'] = df['X_COORD'] * multiplier
        elif 'BUMP_CENTER_X' in df.columns: df['X_VAL'] = df['BUMP_CENTER_X'] * multiplier
        
        # Y Coord
        if 'Y_COORD' in df.columns: df['Y_VAL'] = df['Y_COORD'] * multiplier
        elif 'BUMP_CENTER_Y' in df.columns: df['Y_VAL'] = df['BUMP_CENTER_Y'] * multiplier
        
        # Z Coord (Layer 감지용)
        if 'Z_COORD' in df.columns: df['Z_VAL'] = df['Z_COORD'] * multiplier
        elif 'BUMP_CENTER_Z' in df.columns: df['Z_VAL'] = df['BUMP_CENTER_Z'] * multiplier
        else: df['Z_VAL'] = 0
        
        # 2. 레이어 구분 로직
        if 'LAYER_NUMBER' in df.columns:
            df['L_NUM'] = df['LAYER_NUMBER'].fillna(0).astype(int)
        else:
            # Z값 기반 자동 레이어링
            z_unique = np.sort(df['Z_VAL'].dropna().unique())
            if len(z_unique) > 1:
                diffs = np.diff(z_unique)
                splits = z_unique[1:][diffs > layer_gap]
                l_assign = np.zeros(len(df), dtype=int)
                for s in splits:
                    l_assign[df['Z_VAL'] >= s] += 1
                df['L_NUM'] = l_assign
            else:
                df['L_NUM'] = 0
                
        # 3. 고유 식별자 설정 (Pillar > Group)
        if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
        elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
        else: df['P_ID'] = df.index
        
        df['SOURCE_FILE'] = os.path.splitext(f.name)[0]
        all_dfs.append(df)
        
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# --- [2] Pitch 계산 로직 (Missing Bump 배수 필터 적용) ---
def calculate_pitch(df, pitch_sens):
    grid_size = 1.0 # 정렬 오차 허용 범위
    group_keys = ['SOURCE_FILE', 'L_NUM']
    
    # X-Pitch (동일 행 내부 간격)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['Y_GRID', 'X_VAL'])
    df['ID_STEP'] = df.groupby(group_keys + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = np.where(df['ID_STEP'] == 1, df.groupby(group_keys + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)
    
    # Y-Pitch (동일 열 내부 간격)
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(group_keys + ['X_GRID'])['Y_VAL'].diff().abs()
    
    # 배수 Pitch(Missing) 및 Noise 필터링
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg_p = v.mean()
            # 평균 1.5배 이상(Missing) 또는 0.5배 미만 제거
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
            # IQR 필터 적용
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            df.loc[(df[col] < q1 - pitch_sens * iqr) | (df[col] > q3 + pitch_sens * iqr), col] = np.nan
    return df

# --- [3] UI 구성 ---
st.set_page_config(page_title="NLX Bump Intelligence", layout="wide")
st.title("🔬 NLX Unified Bump Analysis Dashboard")

with st.sidebar:
    st.header("📂 Data Config")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
    multiplier = st.number_input("Unit Multiplier (Int)", value=1, step=1, format="%d", help="mm->um면 1000 입력")
    layer_gap = st.number_input("Layer Detection Gap (um)", value=5.0, step=0.5)
    
    if uploaded_files:
        # 1차 데이터 로드 및 전처리
        combined_df = process_data(uploaded_files, multiplier, layer_gap)
        
        # 2. 분석 지표 선택 (Pivot)
        exclude = ['X_VAL', 'Y_VAL', 'Z_VAL', 'L_NUM', 'P_ID', 'SOURCE_FILE', 'Y_GRID', 'X_GRID', 'ID_STEP', 'X_COORD', 'Y_COORD', 'Z_COORD', 'PILLAR_NUMBER', 'LAYER_NUMBER', 'GROUP_ID']
        available_metrics = [c for c in combined_df.columns if c not in exclude and combined_df[c].dtype in [np.float64, np.int64]]
        target_col = st.selectbox("🎯 Select Target Measurement", available_metrics)
        
        st.markdown("---")
        st.subheader("🧹 Analysis Options")
        use_iqr = st.checkbox("Apply Global IQR Filter", value=True)
        pitch_sens = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2)
        
        st.markdown("---")
        st.subheader("📏 Visualization")
        show_legend = st.checkbox("Show Legend", value=True)
        leg_loc = st.selectbox("Legend Loc", ["best", "upper right", "right", "center"])
        use_manual_axis = st.checkbox("Manual Axis Scale", value=False)
        v_min = st.number_input("Min Limit", value=-10.0)
        v_max = st.number_input("Max Limit", value=10.0)

if uploaded_files and combined_df is not None:
    df = combined_df.copy()
    
    # 타겟 값 배율 적용 및 정제
    df['TARGET_VAL'] = df[target_col] * multiplier
    if use_iqr:
        q1, q3 = df['TARGET_VAL'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df['TARGET_VAL'] >= q1 - 1.5 * iqr) & (df['TARGET_VAL'] <= q3 + 1.5 * iqr)]
    
    # Pitch 계산 수행
    df = calculate_pitch(df, pitch_sens)
    
    # 모드 판별 (Multi-Layer Shift 여부)
    is_multi_mode = 'LAYER_NUMBER' in df.columns and 'PILLAR_NUMBER' in df.columns
    
    # 탭 구성
    tab_labels = ["📊 Statistics", "🗺️ Spatial Heatmap", "🎯 Pitch Analysis", "🧊 3D Stack View"]
    if is_multi_mode:
        tab_labels.insert(2, "📉 Shift Trend")
    tabs = st.tabs(tab_labels)

    # --- Tab 0: Statistics ---
    with tabs[0]:
        sel_layer = st.selectbox("View Layer", ["All"] + [f"L{i}" for i in sorted(df['L_NUM'].unique())])
        plot_df = df if sel_layer == "All" else df[df['L_NUM'] == int(sel_layer[1:])]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(data=plot_df, x='TARGET_VAL', hue='SOURCE_FILE', kde=True, ax=ax)
            if use_manual_axis: ax.set_xlim(v_min, v_max)
            if show_legend: ax.legend(loc=leg_loc, title=None)
            ax.set_title(f"{target_col} Distribution")
            st.pyplot(fig)
        with c2:
            st.markdown(f"**{target_col} Summary (um)**")
            stats = plot_df.groupby(['SOURCE_FILE', 'L_NUM'])['TARGET_VAL'].agg(['mean', 'std', 'count']).reset_index()
            st.dataframe(stats.style.format(precision=3), use_container_width=True)

    # --- Tab 1: Heatmap ---
    with tabs[1]:
        st.subheader(f"Spatial Heatmap: {target_col}")
        fig_heat = px.scatter(df, x='X_VAL', y='Y_VAL', color='TARGET_VAL', facet_col='L_NUM',
                              hover_data=['P_ID'], color_continuous_scale='Viridis')
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Tab: Shift Trend (Multi-Mode 전용) ---
    if is_multi_mode:
        with tabs[2]:
            st.subheader("Layer-to-Layer Shift Trend (Relative to L0)")
            trend_data = []
            for src in df['SOURCE_FILE'].unique():
                src_df = df[df['SOURCE_FILE'] == src]
                base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                for lyr in sorted(df['L_NUM'].unique()):
                    if lyr == 0: continue
                    target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                    m = pd.merge(base, target, on='P_ID', suffixes=('_0', '_L'))
                    if not m.empty:
                        trend_data.append({'Source': src, 'Layer': lyr, 
                                           'DX': (m['X_VAL_L'] - m['X_VAL_0']).mean(),
                                           'DY': (m['Y_VAL_L'] - m['Y_VAL_0']).mean()})
            if trend_data:
                tdf = pd.DataFrame(trend_data)
                col_x, col_y = st.columns(2)
                with col_x:
                    fig_tx = px.line(tdf, x='DX', y='Layer', color='Source', markers=True, title="X-Shift Trend")
                    if use_manual_axis: fig_tx.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_tx, use_container_width=True)
                with col_y:
                    fig_ty = px.line(tdf, x='DY', y='Layer', color='Source', markers=True, title="Y-Shift Trend")
                    if use_manual_axis: fig_ty.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_ty, use_container_width=True)

    # --- Tab: Pitch Analysis ---
    pitch_idx = 3 if is_multi_mode else 2
    with tabs[pitch_idx]:
        st.subheader("🎯 Bump Pitch Distribution (Adjacent Neighbors)")
        cp1, cp2 = st.columns(2)
        with cp1:
            fig_px, ax_px = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='SOURCE_FILE', y='X_PITCH', hue='L_NUM', ax=ax_px)
            ax_px.set_title("X-Pitch (um)")
            st.pyplot(fig_px)
        with cp2:
            fig_py, ax_py = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='SOURCE_FILE', y='Y_PITCH', hue='L_NUM', ax=ax_py)
            ax_py.set_title("Y-Pitch (um)")
            st.pyplot(fig_py)

    # --- Tab: 3D Stack View ---
    with tabs[-1]:
        st.subheader("🧊 Interactive 3D Layer Stack View")
        fig_3d = px.scatter_3d(df, x='X_VAL', y='Y_VAL', z='Z_VAL', color='TARGET_VAL',
                               opacity=0.7, color_continuous_scale='Turbo', hover_data=['P_ID', 'L_NUM'])
        fig_3d.update_layout(height=800)
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("CSV 파일을 업로드하면 자동으로 데이터 형식을 분석하여 대시보드를 생성합니다.")