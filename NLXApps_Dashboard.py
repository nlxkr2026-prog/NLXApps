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
        
        # 1. 좌표 컬럼 탐색 및 표준화 (Multiplier 적용)
        x_col = next((c for c in ['X_COORD', 'BUMP_CENTER_X'] if c in df.columns), None)
        y_col = next((c for c in ['Y_COORD', 'BUMP_CENTER_Y'] if c in df.columns), None)
        z_col = next((c for c in ['Z_COORD', 'BUMP_CENTER_Z', 'INTERSECTION_HEIGHT'] if c in df.columns), None)

        if x_col: df['X_VAL'] = df[x_col] * multiplier
        if y_col: df['Y_VAL'] = df[y_col] * multiplier
        if z_col: df['Z_VAL'] = df[z_col] * multiplier
        
        # 2. 레이어 구분 로직 (명시적 컬럼 우선, 없으면 Z값 기반 자동 감지)
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
            else:
                df['L_NUM'] = 0
                
        # 3. 고유 식별자 설정 (Pillar > Group)
        if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
        elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
        else: df['P_ID'] = df.index
        
        df['ORIG_SOURCE'] = os.path.splitext(f.name)[0]
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

# --- [2] Pitch 계산 로직 ---
def calculate_pitch(df, pitch_sens):
    grid_size = 1.0
    group_keys = ['SOURCE_NAME', 'L_NUM']
    
    # X-Pitch
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['Y_GRID', 'X_VAL'])
    df['ID_STEP'] = df.groupby(group_keys + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = df.groupby(group_keys + ['Y_GRID'])['X_VAL'].diff().abs()
    df.loc[df['ID_STEP'] != 1, 'X_PITCH'] = np.nan
    
    # Y-Pitch
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_keys + ['X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(group_keys + ['X_GRID'])['Y_VAL'].diff().abs()
    
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg_p = v.mean()
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
            q1, q3 = v.quantile([0.25, 0.75])
            df.loc[(df[col] < q1 - pitch_sens * (q3-q1)) | (df[col] > q3 + pitch_sens * (q3-q1)), col] = np.nan
    return df

# --- [3] UI & 대시보드 메인 ---
st.set_page_config(page_title="NLX Analyzer Final Pro", layout="wide")
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
        
        # 분석 지표 자동 추출 (좌표값 포함하여 비교 가능하도록 설정)
        exclude = ['L_NUM', 'P_ID', 'SOURCE_NAME', 'ORIG_SOURCE', 'Y_GRID', 'X_GRID', 'ID_STEP', 'PILLAR_NUMBER', 'LAYER_NUMBER', 'GROUP_ID', 'ID_DIFF', 'X_PITCH', 'Y_PITCH', 'VALUE', 'STATUS']
        available_metrics = [c for c in combined_df.columns if c not in exclude]
        target_col = st.selectbox("🎯 Target Measurement", available_metrics if available_metrics else ["None"])
        
        st.markdown("---")
        st.subheader("🎨 Custom Plot Labels")
        custom_x_name = st.text_input("X-Axis Name", value="X Position (um)")
        custom_y_name = st.text_input("Y-Axis Name", value="Value (um)")
        use_manual_scale = st.checkbox("Enable Manual Scale", value=False)
        v_min, v_max = st.number_input("Min Limit", value=-10.0), st.number_input("Max Limit", value=10.0)

        st.markdown("---")
        st.subheader("🧊 3D Highlight Settings")
        use_outlier_3d = st.checkbox("Highlight 3D Outliers", value=False)
        out_low, out_high = st.number_input("3D Under (Yellow)", value=-5.0), st.number_input("3D Over (Red)", value=5.0)

        st.markdown("---")
        use_iqr = st.checkbox("Apply Global IQR Filter", value=True)
        pitch_sens = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2)
        leg_loc = st.selectbox("Legend Location", ["best", "upper right", "right", "center"])

if uploaded_files and combined_df is not None:
    df = combined_df.copy()
    if target_col != "None":
        df['VALUE'] = df[target_col] * multiplier
        if use_iqr:
            q1, q3 = df['VALUE'].quantile([0.25, 0.75])
            df = df[(df['VALUE'] >= q1 - 1.5*(q3-q1)) & (df['VALUE'] <= q3 + 1.5*(q3-q1))]
    
    df = calculate_pitch(df, pitch_sens)
    
    # 모드 판별 (X_COORD와 Pillar가 모두 존재할 때만 활성화)
    is_multi_shift = 'X_COORD' in df.columns and 'PILLAR_NUMBER' in df.columns and df['L_NUM'].nunique() > 1
    
    tabs = st.tabs(["📊 Statistics", "📈 Comparison", "📉 Shift Trend", "🎯 Pitch Analysis", "🧊 3D View"])

    # --- Tab 0: Statistics ---
    with tabs[0]:
        lyr_sel = st.selectbox("Layer View", ["All"] + [f"Layer {i}" for i in sorted(df['L_NUM'].unique())])
        pdf = df if lyr_sel == "All" else df[df['L_NUM'] == int(lyr_sel.split(" ")[1])]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=pdf, x='VALUE', hue='SOURCE_NAME', kde=True, ax=ax)
        if use_manual_scale: ax.set_xlim(v_min, v_max)
        ax.set_xlabel(custom_x_name); st.pyplot(fig)

    # --- Tab 1: Comparison (복구 확인) ---
    with tabs[1]:
        st.subheader(f"Layer-wise {target_col} Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='L_NUM', y='VALUE', hue='SOURCE_NAME', ax=ax)
        if use_manual_scale: ax.set_ylim(v_min, v_max)
        ax.set_ylabel(custom_y_name); ax.set_xlabel("Layer Number")
        ax.legend(loc=leg_loc); st.pyplot(fig)

    # --- Tab 2: Shift Trend (X축: Shift, Y축: Layer) ---
    with tabs[2]:
        if not is_multi_shift:
            st.warning("Trend 분석은 'X_COORD'와 'PILLAR_NUMBER'가 포함된 다층 데이터 전용입니다.")
        else:
            trend_list = []
            for src in df['SOURCE_NAME'].unique():
                src_df = df[df['SOURCE_NAME'] == src]
                # Layer 0을 기준 좌표로 잡음
                base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                for lyr in sorted(df['L_NUM'].unique()):
                    if lyr == 0: continue
                    target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                    # Pillar Number를 기준으로 병합
                    m = pd.merge(base, target, on='P_ID', suffixes=('_0', '_L'))
                    if not m.empty:
                        trend_list.append({
                            'Source': src, 
                            'Layer': lyr, 
                            'DX': (m['X_VAL_L'] - m['X_VAL_0']).mean(), 
                            'DY': (m['Y_VAL_L'] - m['Y_VAL_0']).mean()
                        })
            if trend_list:
                tdf = pd.DataFrame(trend_list)
                mode = st.radio("View Axis", ["X & Y", "X Only", "Y Only"], horizontal=True)
                fig_trend = px.line(tdf, y='Layer', markers=True, title="Relative Layer Shift Trend from Layer 0")
                if mode in ["X & Y", "X Only"]: fig_trend.add_scatter(x=tdf['DX'], y=tdf['Layer'], name="DX (Shift X)")
                if mode in ["X & Y", "Y Only"]: fig_trend.add_scatter(x=tdf['DY'], y=tdf['Layer'], name="DY (Shift Y)")
                fig_trend.update_layout(xaxis_title="Average Displacement (um)", yaxis_title="Layer Number")
                if use_manual_scale: fig_trend.update_xaxes(range=[v_min, v_max])
                st.plotly_chart(fig_trend, use_container_width=True)

    # --- Tab 3: Pitch Analysis (통계 & 레이어 필터) ---
    with tabs[3]:
        st.subheader("🎯 Bump Pitch Analysis")
        lyr_p = st.selectbox("Select Layer for Pitch", ["All"] + [f"Layer {i}" for i in sorted(df['L_NUM'].unique())])
        p_df = df if lyr_p == "All" else df[df['L_NUM'] == int(lyr_p.split(" ")[1])]
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(); sns.boxplot(data=p_df, x='SOURCE_NAME', y='X_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("X-Pitch (um)"); st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(); sns.boxplot(data=p_df, x='SOURCE_NAME', y='Y_PITCH', hue='L_NUM', ax=ax)
            ax.set_title("Y-Pitch (um)"); st.pyplot(fig)
        
        st.markdown("**Pitch Summary Statistics**")
        st.dataframe(p_df.groupby(['SOURCE_NAME', 'L_NUM'])[['X_PITCH', 'Y_PITCH']].agg(['mean', 'std', 'count']).style.format("{:.3f}"), use_container_width=True)

    # --- Tab 4: 3D View (Highlight) ---
    with tabs[4]:
        st.subheader("🧊 Interactive 3D Stack View")
        plot_3d = df.copy()
        if use_outlier_3d and target_col != "None":
            cond = [(plot_3d['VALUE'] < out_low), (plot_3d['VALUE'] > out_high)]
            plot_3d['Status'] = np.select(cond, ['Under (Yellow)', 'Over (Red)'], default='Normal')
            fig_3d = px.scatter_3d(plot_3d, x='X_VAL', y='Y_VAL', z='Z_VAL', color='Status', color_discrete_map={'Under (Yellow)': 'yellow', 'Over (Red)': 'red', 'Normal': 'blue'})
        else:
            fig_3d = px.scatter_3d(plot_3d, x='X_VAL', y='Y_VAL', z='Z_VAL', color='VALUE', color_continuous_scale='Turbo')
        fig_3d.update_layout(height=800); st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("Upload CSV files to begin.")