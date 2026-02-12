import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- [1] 데이터 전처리 엔진 (mm/um 대응 및 자동 레이어링) ---
def process_data(files, multiplier, layer_gap):
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 좌표 표준화 (Multiplier 적용)
        x_col = next((c for c in ['X_COORD', 'BUMP_CENTER_X'] if c in df.columns), None)
        y_col = next((c for c in ['Y_COORD', 'BUMP_CENTER_Y'] if c in df.columns), None)
        z_col = next((c for c in ['Z_COORD', 'BUMP_CENTER_Z', 'INTERSECTION_HEIGHT'] if c in df.columns), None)

        if x_col: df['X_VAL'] = df[x_col] * multiplier
        else: df['X_VAL'] = 0
        if y_col: df['Y_VAL'] = df[y_col] * multiplier
        else: df['Y_VAL'] = 0
        if z_col: df['Z_VAL'] = df[z_col] * multiplier
        else: df['Z_VAL'] = 0
        
        # 레이어 구분 로직 (Z값 기반 자동 감지 혹은 컬럼 우선)
        if 'LAYER_NUMBER' in df.columns:
            df['L_NUM'] = df['LAYER_NUMBER'].fillna(0).astype(int)
        else:
            z_unique = np.sort(df['Z_VAL'].dropna().unique())
            if len(z_unique) > 1:
                diffs = np.diff(z_unique)
                splits = z_unique[1:][diffs > layer_gap]
                l_assign = np.zeros(len(df), dtype=int)
                for s in splits: l_assign[df['Z_VAL'] >= s] += 1
                df['L_NUM'] = l_assign
            else: df['L_NUM'] = 0
                
        # 고유 식별자 설정 (Pillar > Group)
        if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
        elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
        else: df['P_ID'] = df.index
        
        df['ORIG_SOURCE'] = os.path.splitext(f.name)[0]
        all_dfs.append(df)
        
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# --- [2] Pitch 계산 로직 (Missing Bump 배수 필터 적용) ---
def calculate_pitch(df, pitch_sens):
    grid_size = 1.0 
    group_keys = ['SOURCE_NAME', 'L_NUM']
    
    # X-Pitch (ID 연속성 체크)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['Y_GRID', 'X_VAL'])
    df['ID_STEP'] = df.groupby(group_keys + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = df.groupby(group_keys + ['Y_GRID'])['X_VAL'].diff().abs()
    df.loc[df['ID_STEP'] != 1, 'X_PITCH'] = np.nan
    
    # Y-Pitch
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(group_keys + ['X_GRID'])['Y_VAL'].diff().abs()
    
    # 통계적 필터링
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg_p = v.mean()
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            df.loc[(df[col] < q1 - pitch_sens * iqr) | (df[col] > q3 + pitch_sens * iqr), col] = np.nan
    return df

# --- [3] UI & 대시보드 메인 ---
st.set_page_config(page_title="NLX Bump Analyzer Pro", layout="wide")
st.title("🔬 NLX Unified Bump Analysis Dashboard")

with st.sidebar:
    st.header("📂 Data & Legend Settings")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
    multiplier = st.number_input("Unit Multiplier (Int)", value=1, step=1, format="%d")
    layer_gap = st.number_input("Layer Detection Gap (um)", value=5.0, step=0.5)
    
    custom_names = {}
    if uploaded_files:
        st.subheader("✏️ Legend Names")
        for f in uploaded_files:
            orig = os.path.splitext(f.name)[0]
            custom_names[orig] = st.text_input(f"Name for {f.name}", value=orig)

        combined_df = process_data(uploaded_files, multiplier, layer_gap)
        combined_df['SOURCE_NAME'] = combined_df['ORIG_SOURCE'].map(custom_names)
        
        # 분석 지표 자동 추출
        exclude = ['X_VAL', 'Y_VAL', 'Z_VAL', 'L_NUM', 'P_ID', 'SOURCE_NAME', 'ORIG_SOURCE', 'Y_GRID', 'X_GRID', 'ID_STEP', 'X_COORD', 'Y_COORD', 'Z_COORD', 'PILLAR_NUMBER', 'LAYER_NUMBER', 'GROUP_ID', 'INTERSECTION_HEIGHT', 'ID_DIFF']
        available_metrics = [c for c in combined_df.columns if c not in exclude and combined_df[c].dtype in [np.float64, np.int64]]
        target_col = st.selectbox("🎯 Target Measurement", available_metrics if available_metrics else ["None"])
        
        st.markdown("---")
        st.subheader("🎨 Custom Labels")
        custom_x_name = st.text_input("X-Axis Name (Plot)", value="X Position (um)")
        custom_y_name = st.text_input("Y-Axis Name (Plot)", value="Measurement Value (um)")

        st.markdown("---")
        st.subheader("📏 Manual Scale Control")
        use_manual_scale = st.checkbox("Apply Manual Min/Max", value=False)
        v_min = st.number_input("Axis/Color Min", value=-10.0)
        v_max = st.number_input("Axis/Color Max", value=10.0)

        st.markdown("---")
        use_iqr = st.checkbox("Apply Global IQR Filter", value=True)
        pitch_sens = st.slider("Pitch Sensitivity", 0.5, 3.0, 1.2)
        leg_loc = st.selectbox("Legend Location", ["best", "upper right", "right", "center"])

if uploaded_files and combined_df is not None:
    df = combined_df.copy()
    
    # 1. 타겟 데이터 스케일링 및 정제
    df['VALUE'] = df[target_col] * multiplier
    if use_iqr:
        q1, q3 = df['VALUE'].quantile([0.25, 0.75])
        df = df[(df['VALUE'] >= q1 - 1.5*(q3-q1)) & (df['VALUE'] <= q3 + 1.5*(q3-q1))]
    
    # 2. Pitch 계산
    df = calculate_pitch(df, pitch_sens)
    
    # 3. 모드 판별 (X_COORD 및 Pillar 기반 Shift Trend 조건)
    is_multi_shift = 'X_COORD' in df.columns and 'PILLAR_NUMBER' in df.columns and df['L_NUM'].nunique() > 1
    
    # 탭 구성 (Comparison 포함)
    tab_list = ["📊 Statistics", "📈 Layer Comparison", "🗺️ Spatial Map", "🎯 Pitch Analysis", "🧊 3D View"]
    if is_multi_shift: tab_list.insert(2, "📉 Shift Trend")
    tabs = st.tabs(tab_list)

    # --- Tab 0: Statistics ---
    with tabs[0]:
        sel_layer = st.selectbox("Select View Layer", ["All"] + [f"Layer {i}" for i in sorted(df['L_NUM'].unique())])
        plot_df = df if sel_layer == "All" else df[df['L_NUM'] == int(sel_layer.split(" ")[1])]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=plot_df, x='VALUE', hue='SOURCE_NAME', kde=True, ax=ax)
            ax.set_xlabel(custom_x_name); ax.set_title(f"{target_col} Distribution ({sel_layer})")
            if use_manual_scale: ax.set_xlim(v_min, v_max)
            ax.legend(loc=leg_loc)
            st.pyplot(fig)
        with c2:
            st.markdown(f"**{target_col} Stats**")
            st.dataframe(plot_df.groupby(['SOURCE_NAME', 'L_NUM'])['VALUE'].agg(['mean', 'std', 'count']).style.format("{:.3f}"), use_container_width=True)

    # --- Tab 1: Layer Comparison ---
    with tabs[1]:
        st.subheader(f"Layer-wise {target_col} Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='L_NUM', y='VALUE', hue='SOURCE_NAME', ax=ax)
        ax.set_ylabel(custom_y_name); ax.set_xlabel("Layer Number")
        if use_manual_scale: ax.set_ylim(v_min, v_max)
        ax.legend(loc=leg_loc)
        st.pyplot(fig)

    # --- Tab 2: Shift Trend (X_COORD & Pillar 기반) ---
    if is_multi_shift:
        with tabs[2]:
            st.subheader("📉 Layer-to-Layer Shift Trend (Base: L0)")
            trend_list = []
            for src in df['SOURCE_NAME'].unique():
                src_df = df[df['SOURCE_NAME'] == src]
                base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                for lyr in sorted(df['L_NUM'].unique()):
                    if lyr == 0: continue
                    target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                    m = pd.merge(base, target, on='P_ID', suffixes=('_0', '_L'))
                    if not m.empty:
                        trend_list.append({'Source': src, 'Layer': lyr, 'DX': (m['X_VAL_L'] - m['X_VAL_0']).mean(), 'DY': (m['Y_VAL_L'] - m['Y_VAL_0']).mean()})
            if trend_list:
                tdf = pd.DataFrame(trend_list)
                ctx, cty = st.columns(2)
                with ctx:
                    fig_tx = px.line(tdf, x='DX', y='Layer', color='Source', markers=True, title="X-Shift Trend")
                    if use_manual_scale: fig_tx.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_tx, use_container_width=True)
                with cty:
                    fig_ty = px.line(tdf, x='DY', y='Layer', color='Source', markers=True, title="Y-Shift Trend")
                    if use_manual_scale: fig_ty.update_xaxes(range=[v_min, v_max])
                    st.plotly_chart(fig_ty, use_container_width=True)

    # --- Tab: Spatial Map (Heatmap) ---
    spatial_idx = 3 if is_multi_shift else 2
    with tabs[spatial_idx]:
        fig_heat = px.scatter(df, x='X_VAL', y='Y_VAL', color='VALUE', facet_col='L_NUM', 
                              color_continuous_scale='Viridis', labels={'X_VAL': custom_x_name, 'Y_VAL': 'Y (um)'})
        if use_manual_scale: fig_heat.update_coloraxes(cmin=v_min, cmax=v_max)
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Tab: Pitch Analysis ---
    pitch_idx = 4 if is_multi_shift else 3
    with tabs[pitch_idx]:
        st.subheader("🎯 Bump Pitch Analysis")
        cp1, cp2 = st.columns(2)
        with cp1:
            fig, ax = plt.subplots(); sns.boxplot(data=df, x='SOURCE_NAME', y='X_PITCH', hue='L_NUM', ax=ax)
            ax.set_ylabel("X-Pitch (um)"); st.pyplot(fig)
        with cp2:
            fig, ax = plt.subplots(); sns.boxplot(data=df, x='SOURCE_NAME', y='Y_PITCH', hue='L_NUM', ax=ax)
            ax.set_ylabel("Y-Pitch (um)"); st.pyplot(fig)

    # --- Tab: 3D Stack View (완벽 복구) ---
    with tabs[-1]:
        st.subheader("🧊 Interactive 3D Layer Stack View")
        fig_3d = px.scatter_3d(df, x='X_VAL', y='Y_VAL', z='Z_VAL', color='VALUE',
                               opacity=0.7, color_continuous_scale='Turbo',
                               labels={'X_VAL': 'X (um)', 'Y_VAL': 'Y (um)', 'Z_VAL': 'Height (Z)', 'VALUE': target_col},
                               height=800)
        if use_manual_scale: fig_3d.update_coloraxes(cmin=v_min, cmax=v_max)
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("Upload CSV files to begin analysis.")