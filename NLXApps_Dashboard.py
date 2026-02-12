import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import os

# --- [1] 데이터 전처리 및 정밀 Pitch 계산 로직 ---
def process_data(df, scale_factor, apply_iqr, pitch_sensitivity):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별
    d_type = None
    target_cols = []
    if 'HEIGHT' in df.columns: d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns: d_type, target_cols = "Radius", ['RADIUS']
    elif 'SHIFT_NORM' in df.columns or 'SHIFT_X' in df.columns: 
        d_type = "Shift"
        for c in ['SHIFT_NORM', 'SHIFT_X', 'SHIFT_Y']:
            if c in df.columns: target_cols.append(c)
    elif 'X_COORD' in df.columns: d_type, target_cols = "Coordinate", ['X_COORD']
    else: return None, None

    # 2. 좌표 설정 (Multiplier 적용)
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    for col in target_cols:
        df[col + '_UM'] = df[col] * scale_factor
    df['MEAS_VALUE'] = df[target_cols[0] + '_UM']

    # 3. 레이어 설정
    if 'LAYER_NUMBER' in df.columns: df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        if len(z_vals) > 1:
            z_diffs = np.diff(z_vals)
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            l_assign = np.zeros(len(df), dtype=int)
            for p in splits: l_assign[df['BUMP_CENTER_Z'] >= p] += 1
            df['L_NUM'] = l_assign
        else: df['L_NUM'] = 0
    else: df['L_NUM'] = 0

    # 4. [정밀 Pitch 알고리즘]
    df['P_ID'] = df['GROUP_ID'] if 'GROUP_ID' in df.columns else df.index
    group_base = ['SOURCE_FILE', 'L_NUM'] if 'SOURCE_FILE' in df.columns else ['L_NUM']
    grid_size = 0.5 

    # X_Pitch 계산 (Y 그리드 기준)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['Y_GRID', 'X_VAL'])
    df['ID_DIFF'] = df.groupby(group_base + ['Y_GRID'])['P_ID'].diff()
    # [제안 반영] ID가 1 차이가 아니면(Missing 발생 지점) Pitch 계산 제외
    df['X_PITCH'] = np.where(df['ID_DIFF'] == 1, df.groupby(group_base + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)

    # Y_Pitch 계산 (X 그리드 기준)
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['X_GRID', 'Y_VAL'])
    # Y축은 ID가 불규칙할 수 있으므로 먼저 Raw 계산 후 통계 필터 적용
    df['Y_P_RAW'] = df.groupby(group_base + ['X_GRID'])['Y_VAL'].diff().abs()

    # [제안 반영] Pitch 통계 필터 (평균 기반 1.5배 이상/이하 제거)
    for col in ['X_PITCH', 'Y_P_RAW']:
        valid_data = df[col].dropna()
        if not valid_data.empty:
            avg_p = valid_data.mean()
            # 평균의 1.5배 초과(Missing 배수) 혹은 0.5배 미만(노이즈) 제거
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
    
    df['Y_PITCH'] = df['Y_P_RAW']

    # Pitch 전용 IQR 필터 (마지막 정제)
    for col in ['X_PITCH', 'Y_PITCH']:
        valid_p = df[col].dropna()
        if not valid_p.empty:
            q1, q3 = valid_p.quantile([0.25, 0.75])
            iqr_p = q3 - q1
            df.loc[(df[col] < q1 - pitch_sensitivity*iqr_p) | (df[col] > q3 + pitch_sensitivity*iqr_p), col] = np.nan

    # 5. 측정값 이상치 제거
    df_clean = df[df['MEAS_VALUE'] != 0].copy()
    if apply_iqr and d_type != "Coordinate":
        qh1, qh3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
        iqr_h = qh3 - qh1
        df_clean = df_clean[(df_clean['MEAS_VALUE'] >= qh1 - 1.5*iqr_h) & (df_clean['MEAS_VALUE'] <= qh3 + 1.5*iqr_h)]

    return df_clean, d_type

# --- 범례 관리 함수 ---
def apply_global_legend(ax, loc, show_legend):
    if not show_legend:
        leg = ax.get_legend()
        if leg: leg.remove()
        return
    try:
        sns.move_legend(ax, loc=loc, title=None)
    except:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles=handles, labels=labels, loc=loc, title=None)

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Professional", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

with st.sidebar:
    st.header("📁 Data Config")
    uploaded_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
    scale = st.number_input("Multiplier (Scale Factor)", value=1.0, format="%.4f")
    use_iqr = st.checkbox("Apply IQR Filter (Meas. Value)", value=True)

    st.markdown("---")
    st.subheader("🎯 Pitch Filter Settings")
    # Pitch 필터 강도 조절 (기본 1.2로 타이트하게 설정)
    pitch_sensitivity = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2, 0.1)

    with st.expander("🎨 Plot Settings", expanded=True):
        p_w = st.slider("Plot Width", 5, 25, 12)
        p_h = st.slider("Plot Height", 3, 15, 6)
        custom_title = st.text_input("Graph Title", "Alignment Analysis")
        x_lbl = st.text_input("X Axis Label", "X Position (um)")
        y_lbl = st.text_input("Y Axis Label", "Y Position (um)")

    with st.expander("📏 Legend & Scale Control", expanded=False):
        show_legend = st.checkbox("Show Legend", value=True)
        global_legend_loc = st.selectbox("Legend Loc", options=["best", "upper right", "upper left", "lower left", "lower right", "right", "center"], index=1)
        use_custom_scale = st.checkbox("Manual Axis Range", value=False)
        v_min = st.number_input("Min Limit", value=-10.0)
        v_max = st.number_input("Max Limit", value=10.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        # pitch_sensitivity 파라미터 전달
        p_df, d_type = process_data(raw_df, scale, use_iqr, pitch_sensitivity)
        if p_df is not None:
            p_df['SOURCE_FILE'] = os.path.splitext(file.name)[0]
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())

        st.markdown("### 📋 Quick Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global Average", f"{combined_df['MEAS_VALUE'].mean():.3f}")
        m2.metric("Global 3-Sigma", f"{(combined_df['MEAS_VALUE'].std()*3):.3f}")
        m3.metric("Max-Min Range", f"{(combined_df['MEAS_VALUE'].max()-combined_df['MEAS_VALUE'].min()):.3f}")
        m4.metric("Total Bumps", f"{len(combined_df):,}")
        
        st.markdown("---")
        tabs = st.tabs(["📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View", "🎯 Pitch Analysis"])

        with tabs[0]:
            sel_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            disp_df = combined_df if sel_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(sel_layer.split(" ")[1])]
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sns.histplot(data=disp_df, x='MEAS_VALUE', hue='SOURCE_FILE', kde=True, ax=ax1)
            if use_custom_scale: ax1.set_xlim(v_min, v_max)
            apply_global_legend(ax1, global_legend_loc, show_legend)
            st.pyplot(fig1)

        with tabs[4]:
            st.subheader("🎯 Pitch Distribution Analysis")
            sel_layer_p = st.selectbox("Select Layer for Pitch", ["All Layers"] + [f"Layer {i}" for i in unique_layers], key="pitch_tab_layer")
            pitch_df = combined_df if sel_layer_p == "All Layers" else combined_df[combined_df['L_NUM'] == int(sel_layer_p.split(" ")[1])]
            
            col_p1, col_p2 = st.columns(2)
            for col, p_type, p_color in zip([col_p1, col_p2], ['X_PITCH', 'Y_PITCH'], ['Blues', 'Reds']):
                with col:
                    st.markdown(f"**{p_type} Analysis**")
                    fig_p, ax_p = plt.subplots(figsize=(p_w/2, p_h))
                    sns.boxplot(data=pitch_df, x='SOURCE_FILE', y=p_type, hue='SOURCE_FILE', ax=ax_p, palette=p_color)
                    apply_global_legend(ax_p, global_legend_loc, show_legend); st.pyplot(fig_p)
                    
                    fig_h, ax_h = plt.subplots(figsize=(p_w/2, p_h))
                    sns.histplot(data=pitch_df, x=p_type, hue='SOURCE_FILE', kde=True, ax=ax_h)
                    apply_global_legend(ax_h, global_legend_loc, show_legend); st.pyplot(fig_h)
            
            st.markdown("**Pitch Stats Summary**")
            st.dataframe(pitch_df.groupby('SOURCE_FILE')[['X_PITCH', 'Y_PITCH']].mean().style.format("{:.3f}"), use_container_width=True)

else:
    st.info("Please upload CSV files.")
    