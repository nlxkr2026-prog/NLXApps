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
        # X/Y/Z 좌표 컬럼 찾기 (X_COORD, BUMP_CENTER_X 등 모두 대응)
        x_col = next((c for c in ['X_COORD', 'BUMP_CENTER_X'] if c in df.columns), None)
        y_col = next((c for c in ['Y_COORD', 'BUMP_CENTER_Y'] if c in df.columns), None)
        z_col = next((c for c in ['Z_COORD', 'BUMP_CENTER_Z', 'INTERSECTION_HEIGHT'] if c in df.columns), None)

        df['X_VAL'] = df[x_col] * multiplier if x_col else 0
        df['Y_VAL'] = df[y_col] * multiplier if y_col else 0
        df['Z_VAL'] = df[z_col] * multiplier if z_col else 0
        
        # 2. 레이어 구분 로직 (Z값 기반 자동 감지)
        if 'LAYER_NUMBER' in df.columns:
            df['L_NUM'] = df['LAYER_NUMBER'].fillna(0).astype(int)
        else:
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
        
        df['ORIG_SOURCE'] = os.path.splitext(f.name)[0]
        all_dfs.append(df)
        
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# --- [2] Pitch 계산 로직 (Missing Bump 고려) ---
def calculate_pitch(df, pitch_sens):
    grid_size = 1.0 # 정렬 오차 허용 그리드
    group_keys = ['SOURCE_NAME', 'L_NUM']
    
    # X-Pitch (Y 그리드 내 정렬)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['Y_GRID', 'X_VAL'])
    df['ID_STEP'] = df.groupby(group_keys + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = df.groupby(group_keys + ['Y_GRID'])['X_VAL'].diff().abs()
    df.loc[df['ID_STEP'] != 1, 'X_PITCH'] = np.nan # 연속되지 않으면 삭제
    
    # Y-Pitch (X 그리드 내 정렬)
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(group_keys + ['X_GRID'])['Y_VAL'].diff().abs()
    
    # 배수 Pitch 제거 및 IQR 필터링
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg_p = v.mean()
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
            q1, q3 = v.quantile([0.25, 0.75])
            df.loc[(df[col] < q1 - pitch_sens * (q3-q1)) | (df[col] > q3 + pitch_sens * (q3-q1)), col] = np.nan
    return df

# --- [3] UI 구성 ---
st.set_page_config(page_title="NLX Bump Analyzer", layout="wide")
st.title("🔬 NLX Unified Bump Analysis Dashboard")

with st.sidebar:
    st.header("📂 Data & Legend Settings")
    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)
    multiplier = st.number_input("Unit Multiplier (Int)", value=1, step=1, format="%d")
    layer_gap = st.number_input("Layer Detection Gap (um)", value=5.0, step=0.5)
    
    custom_names = {}
    if uploaded_files:
        st.subheader("✏️ Edit Legend Names")
        for f in uploaded_files:
            orig = os.path.splitext(f.name)[0]
            custom_names[orig] = st.text_input(f"Name for {f.name}", value=orig)

        combined_df = process_data(uploaded_files, multiplier, layer_gap)
        combined_df['SOURCE_NAME'] = combined_df['ORIG_SOURCE'].map(custom_names)
        
        # 분석 대상 컬럼 선택 (Pivot) - Bump Inner Shift 포함
        exclude = ['X_VAL', 'Y_VAL', 'Z_VAL', 'L_NUM', 'P_ID', 'SOURCE_NAME', 'ORIG_SOURCE', 'Y_GRID', 'X_GRID', 'ID_STEP', 'X_COORD', 'Y_COORD', 'Z_COORD', 'PILLAR_NUMBER', 'LAYER_NUMBER', 'GROUP_ID', 'INTERSECTION_HEIGHT']
        available_metrics = [c for c in combined_df.columns if c not in exclude and combined_df[c].dtype in [np.float64, np.int64]]
        target_col = st.selectbox("🎯 Target Measurement (Shift/Height/Radius)", available_metrics)
        
        st.markdown("---")
        use_iqr = st.checkbox("Apply Global IQR Filter", value=True)
        pitch_sens = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2)
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
    
    # 3. 탭 구성
    is_multi_shift = 'X_COORD' in df.columns and 'PILLAR_NUMBER' in df.columns
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
            ax.set_title(f"{target_col} Distribution ({sel_layer})")
            ax.legend(loc=leg_loc, labels=custom_names.values())
            st.pyplot(fig)
        with c2:
            st.markdown(f"**{target_col} Statistics**")
            stats = plot_df.groupby(['SOURCE_NAME', 'L_NUM'])['VALUE'].agg(['mean', 'std', 'count'])
            st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)

    # --- Tab 1: Layer Comparison (복구된 기능) ---
    with tabs[1]:
        st.subheader(f"Layer-wise {target_col} Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='L_NUM', y='VALUE', hue='SOURCE_NAME', ax=ax)
        ax.set_title(f"{target_col} by Layer Number")
        ax.legend(loc=leg_loc)
        st.pyplot(fig)

    # --- Tab 2: Spatial Map (Heatmap) ---
    with tabs[2 if not is_multi_shift else 3]:
        fig_heat = px.scatter(df, x='X_VAL', y='Y_VAL', color='VALUE', facet_col='L_NUM', 
                              title=f"Spatial Map: {target_col}", color_continuous_scale='Viridis',
                              labels={'X_VAL': 'X (um)', 'Y_VAL': 'Y (um)'})
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Tab: Shift Trend (Multi-Layer Shift 모드) ---
    if is_multi_shift:
        with tabs[2]:
            st.subheader("Layer-to-Layer Global Shift Trend")
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
                c_x, c_y = st.columns(2)
                with c_x: st.plotly_chart(px.line(tdf, x='DX', y='Layer', color='Source', markers=True, title="Avg X-Shift"), use_container_width=True)
                with c_y: st.plotly_chart(px.line(tdf, x='DY', y='Layer', color='Source', markers=True, title="Avg Y-Shift"), use_container_width=True)

    # --- Tab: Pitch Analysis ---
    with tabs[3 if not is_multi_shift else 4]:
        st.subheader("🎯 Bump Pitch Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='SOURCE_NAME', y='X_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("X-Pitch (um)")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='SOURCE_FILE', y='Y_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("Y-Pitch (um)")
            st.pyplot(fig)

    # --- Tab 4: 3D View ---
    with tabs[-1]:
        fig_3d = px.scatter_3d(df, x='X_VAL', y='Y_VAL', z='Z_VAL', color='VALUE',
                               opacity=0.7, color_continuous_scale='Turbo', title=f"3D Stacked: {target_col}")
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("CSV 파일을 업로드하면 자동으로 분석 모드를 설정합니다.")