import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import os

# --- [1] 데이터 전처리 및 분석 로직 ---
def process_data(df, scale_factor, apply_iqr, pitch_sensitivity, layer_gap):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 판별 (Trend는 전용 데이터일 때만)
    is_trend_compatible = 'X_COORD' in df.columns and 'PILLAR_NUMBER' in df.columns
    d_type = None
    target_cols = []
    shift_cols = [c for c in ['SHIFT_NORM', 'SHIFT_X', 'SHIFT_Y'] if c in df.columns]
    
    if shift_cols:
        d_type, target_cols = "Shift", shift_cols
    elif 'HEIGHT' in df.columns:
        d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns:
        d_type, target_cols = "Radius", ['RADIUS']
    else:
        return None, "Unknown", False

    # 2. 좌표 및 측정값 설정 (정수 배율 적용)
    s_val = int(scale_factor)
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * s_val
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * s_val
    
    for col in target_cols:
        df[col + '_UM'] = df[col] * s_val
    
    main_col = 'SHIFT_NORM' if 'SHIFT_NORM' in df.columns else target_cols[0]
    df['MEAS_VALUE'] = df[main_col + '_UM']

    # 3. 레이어 설정 (Scaled Z 기반)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].fillna(0).astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        df['Z_VAL_UM'] = df['BUMP_CENTER_Z'] * s_val
        z_array = np.sort(df['Z_VAL_UM'].unique())
        if len(z_array) > 1:
            z_diffs = np.diff(z_array)
            splits = z_array[1:][z_diffs > layer_gap]
            l_assign = np.zeros(len(df), dtype=int)
            for split in splits: l_assign[df['Z_VAL_UM'] >= split] += 1
            df['L_NUM'] = l_assign
        else: df['L_NUM'] = 0
    else:
        df['L_NUM'] = 0

    # 4. 식별자 및 Pitch 계산 (Missing 지점 제거 포함)
    df['P_ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else df.get('GROUP_ID', df.index)
    grid_size = 0.5
    group_base = ['L_NUM']
    
    # X-Pitch
    df = df.sort_values(by=group_base + ['Y_VAL', 'X_VAL'])
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df['ID_DIFF'] = df.groupby(group_base + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = np.where(df['ID_DIFF'] == 1, df.groupby(group_base + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)
    
    # Y-Pitch
    df = df.sort_values(by=group_base + ['X_VAL', 'Y_VAL'])
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df['Y_PITCH'] = df.groupby(group_base + ['X_GRID'])['Y_VAL'].diff().abs()

    # Pitch 정밀 필터링 (평균 1.5배 필터 + 가변 IQR)
    for col in ['X_PITCH', 'Y_PITCH']:
        v = df[col].dropna()
        if not v.empty:
            avg = v.mean()
            df[col] = np.where((df[col] > avg*1.5) | (df[col] < avg*0.5), np.nan, df[col])
            q1, q3 = v.quantile([0.25, 0.75])
            df.loc[(df[col] < q1 - pitch_sensitivity*(q3-q1)) | (df[col] > q3 + pitch_sensitivity*(q3-q1)), col] = np.nan

    # 5. 측정값 IQR 필터
    df_clean = df[df['MEAS_VALUE'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
        df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5*(q3-q1)) & (df_clean['MEAS_VALUE'] <= q3 + 1.5*(q3-q1))]

    return df_clean, d_type, is_trend_compatible

# --- 범례 제어 함수 ---
def apply_legend_custom(ax, show, loc):
    if not show:
        leg = ax.get_legend()
        if leg: leg.remove()
    else:
        try:
            sns.move_legend(ax, loc=loc, title=None)
        except:
            ax.legend(loc=loc, title=None)

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

with st.sidebar:
    st.header("📁 Config")
    uploaded_files = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=True)
    scale = st.number_input("Multiplier (Int)", value=1, step=1, format="%d")
    layer_gap = st.number_input("Layer Gap (um)", value=5.0, step=0.5)
    use_iqr = st.checkbox("Apply IQR Filter", value=True)
    pitch_sens = st.slider("Pitch Sensitivity", 0.5, 3.0, 1.2)
    
    with st.expander("🎨 Visualization Settings"):
        p_w, p_h = st.slider("Width", 5, 25, 12), st.slider("Height", 3, 15, 6)
        show_legend = st.checkbox("Show Legend", value=True)
        g_loc = st.selectbox("Legend Loc", ["best", "upper right", "upper left", "lower left", "right", "center"], index=1)
        use_manual_scale = st.checkbox("Manual Axis Range", value=False)
        v_min, v_max = st.number_input("Min", value=-10.0), st.number_input("Max", value=10.0)

    with st.expander("🧊 3D Settings"):
        use_outlier_3d = st.checkbox("Highlight 3D Outliers")
        out_low, out_high = st.number_input("Lower Bound", value=-5.0), st.number_input("Upper Bound", value=5.0)

if uploaded_files:
    all_data = []
    has_trend = False
    for f in uploaded_files:
        p_df, d_t, is_t = process_data(pd.read_csv(f), scale, use_iqr, pitch_sens, layer_gap)
        if p_df is not None:
            p_df['SOURCE_FILE'] = os.path.splitext(f.name)[0]
            all_data.append(p_df)
            if is_t: has_trend = True

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        lyrs = sorted(combined_df['L_NUM'].unique())
        ts = st.tabs(["📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View", "🎯 Pitch Analysis"])

        with ts[0]: # Single
            sl = st.selectbox("Select Layer", ["All"] + [f"Layer {i}" for i in lyrs])
            df_s = combined_df if sl == "All" else combined_df[combined_df['L_NUM'] == int(sl.split(" ")[1])]
            fig, ax = plt.subplots(figsize=(p_w, p_h))
            sns.histplot(data=df_s, x='MEAS_VALUE', hue='SOURCE_FILE', kde=True, ax=ax)
            if use_manual_scale: ax.set_xlim(v_min, v_max)
            apply_legend_custom(ax, show_legend, g_loc)
            st.pyplot(fig)

        with ts[1]: # Comparison
            fig, ax = plt.subplots(figsize=(p_w, p_h))
            sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax)
            if use_manual_scale: ax.set_ylim(v_min, v_max)
            apply_legend_custom(ax, show_legend, g_loc)
            st.pyplot(fig)

        with ts[2]: # Trend
            if not has_trend: st.warning("⚠️ Shift Trend data (X_COORD & PILLAR_NUMBER) required.")
            else:
                trend_list = []
                for src in combined_df['SOURCE_FILE'].unique():
                    src_df = combined_df[combined_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    if base.empty: continue
                    for lyr in lyrs:
                        if lyr == 0: continue
                        target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                        merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                        if not merged.empty:
                            trend_list.append({'Source': src, 'Layer': lyr, 'DX': (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean(), 'DY': (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()})
                if trend_list:
                    t_df = pd.DataFrame(trend_list)
                    fig, ax = plt.subplots(figsize=(p_w, p_h))
                    for s in t_df['Source'].unique():
                        d = t_df[t_df['Source'] == s].sort_values('Layer')
                        ax.plot(d['DX'], d['Layer'], marker='o', label=f"{s} (DX)")
                        ax.plot(d['DY'], d['Layer'], marker='s', ls='--', label=f"{s} (DY)")
                    if use_manual_scale: ax.set_xlim(v_min, v_max)
                    ax.legend(); st.pyplot(fig)

        with ts[3]: # 3D View
            if use_outlier_3d:
                cond = [(combined_df['MEAS_VALUE'] < out_low), (combined_df['MEAS_VALUE'] > out_high)]
                combined_df['Status'] = np.select(cond, ['Under (Yellow)', 'Over (Red)'], default='Normal')
                fig3d = px.scatter_3d(combined_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='Status', color_discrete_map={'Under (Yellow)': 'yellow', 'Over (Red)': 'red', 'Normal': 'blue'})
            else:
                fig3d = px.scatter_3d(combined_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='MEAS_VALUE', color_continuous_scale='viridis')
            fig3d.update_layout(height=700); st.plotly_chart(fig3d, use_container_width=True)

        with ts[4]: # Pitch
            c1, c2 = st.columns(2)
            for col, p_type, p_color in zip([c1, c2], ['X_PITCH', 'Y_PITCH'], ['Blues', 'Reds']):
                with col:
                    fig, ax = plt.subplots(figsize=(p_w/2, p_h))
                    sns.boxplot(data=combined_df, x='SOURCE_FILE', y=p_type, hue='SOURCE_FILE', ax=ax, palette=p_color)
                    apply_legend_custom(ax, show_legend, g_loc)
                    st.pyplot(fig)
else:
    st.info("Please upload CSV files.")