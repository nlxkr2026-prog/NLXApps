import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import os

# --- [1] 데이터 전처리 및 정밀 Pitch 계산 로직 ---
def process_data(df, scale_factor, apply_iqr):
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

    # 2. 좌표 설정 (Multiplier를 즉시 적용하여 모든 계산을 um 단위로 통일)
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

    # 4. [개선] 양산형 Pitch 계산 (데이터 스케일 대응형)
    df['P_ID'] = df['GROUP_ID'] if 'GROUP_ID' in df.columns else df.index
    group_base = ['SOURCE_FILE', 'L_NUM'] if 'SOURCE_FILE' in df.columns else ['L_NUM']

    # 데이터의 최소 간격을 파악하여 그리드 크기를 유동적으로 결정 (um 단위 기준 보통 1.0~10.0 사이)
    # scale_factor가 1000이면 round(0), 1이면 round(3)이 적절함. 이를 자동화하기 위해 0.5um 단위 그리드 사용
    grid_size = 0.5 
    
    # X_Pitch 계산 (Y 그리드 기준 정렬)
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['Y_GRID', 'X_VAL'])
    
    # GroupID가 연속적인 경우만 계산 (행 바뀜 방지)
    df['ID_DIFF'] = df.groupby(group_base + ['Y_GRID'])['P_ID'].diff()
    df['X_PITCH'] = np.where(df['ID_DIFF'] == 1, df.groupby(group_base + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)

    # Y_Pitch 계산 (X 그리드 기준 정렬)
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['X_GRID', 'Y_VAL'])
    
    # Y축 방향은 ID가 건너뛸 수 있으므로 좌표 차이가 일정 거리(예: 예상 Pitch의 1.5배) 이내인 경우만 계산
    df['Y_P_RAW'] = df.groupby(group_base + ['X_GRID'])['Y_VAL'].diff().abs()
    # 일반적인 반도체 범프 피치(약 50um)를 고려하여 100um 이상은 무시
    df['Y_PITCH'] = np.where(df['Y_P_RAW'] < 100, df['Y_P_RAW'], np.nan)

    # Pitch 이상치 제거 (IQR)
    for col in ['X_PITCH', 'Y_PITCH']:
        valid_p = df[col].dropna()
        if not valid_p.empty:
            q1, q3 = valid_p.quantile([0.25, 0.75])
            iqr_p = q3 - q1
            df.loc[(df[col] < q1 - 1.5*iqr_p) | (df[col] > q3 + 1.5*iqr_p), col] = np.nan

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
    scale = st.number_input("Multiplier (Scale Factor)", value=1.0, format="%.4f", step=None)
    use_iqr = st.checkbox("Apply IQR Filter", value=True)

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
        v_min = st.number_input("Min Limit", value=-10.0, step=None)
        v_max = st.number_input("Max Limit", value=10.0, step=None)

    with st.expander("🧊 3D & Outlier Settings", expanded=False):
        color_option = st.selectbox("Color Theme", ["Viridis", "Plasma", "Jet", "Turbo"])
        use_outlier_filter = st.checkbox("Highlight Outliers")
        outlier_low = st.number_input("Lower Bound (Yellow)", value=-5.0, step=None)
        outlier_high = st.number_input("Upper Bound (Red)", value=5.0, step=None)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        # Multiplier 적용 로직이 포함된 process_data 호출
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE_FILE'] = os.path.splitext(file.name)[0]
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())

        # Quick Summary
        st.markdown("### 📋 Quick Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global Average", f"{combined_df['MEAS_VALUE'].mean():.3f}")
        m2.metric("Global 3-Sigma", f"{(combined_df['MEAS_VALUE'].std()*3):.3f}")
        m3.metric("Max-Min Range", f"{(combined_df['MEAS_VALUE'].max()-combined_df['MEAS_VALUE'].min()):.3f}")
        m4.metric("Total Bumps", f"{len(combined_df):,}")

        tabs = st.tabs(["📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View", "🎯 Pitch Analysis"])

        with tabs[0]:
            sel_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            disp_df = combined_df if sel_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(sel_layer.split(" ")[1])]
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sns.histplot(data=disp_df, x='MEAS_VALUE', hue='SOURCE_FILE', kde=True, ax=ax1)
            if use_custom_scale: ax1.set_xlim(v_min, v_max)
            apply_global_legend(ax1, global_legend_loc, show_legend)
            ax1.set_title(custom_title); ax1.set_xlabel(x_lbl); ax1.set_ylabel(y_lbl)
            st.pyplot(fig1)

        with tabs[3]:
            st.subheader("Interactive 3D Layer Stack View")
            plot_3d_df = combined_df.copy()
            if use_outlier_filter:
                conditions = [(plot_3d_df['MEAS_VALUE'] < outlier_low), (plot_3d_df['MEAS_VALUE'] > outlier_high)]
                choices = ['Under Limit (Low)', 'Over Limit (High)']
                plot_3d_df['Status'] = np.select(conditions, choices, default='Normal')
                fig_3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='Status',
                                     color_discrete_map={'Under Limit (Low)': 'yellow', 'Over Limit (High)': 'red', 'Normal': 'blue'},
                                     opacity=0.6)
            else:
                fig_3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='MEAS_VALUE', color_continuous_scale=color_option.lower())
            fig_3d.update_layout(height=700); st.plotly_chart(fig_3d, use_container_width=True)

        with tabs[4]:
            st.subheader("🎯 Pitch Analysis (X & Y Distribution)")
            col_p1, col_p2 = st.columns(2)
            for col, p_type, p_color in zip([col_p1, col_p2], ['X_PITCH', 'Y_PITCH'], ['Blues', 'Reds']):
                with col:
                    st.markdown(f"**{p_type} Analysis**")
                    fig_p, ax_p = plt.subplots(figsize=(p_w/2, p_h))
                    sns.boxplot(data=combined_df, x='SOURCE_FILE', y=p_type, hue='SOURCE_FILE', ax=ax_p, palette=p_color)
                    apply_global_legend(ax_p, global_legend_loc, show_legend); st.pyplot(fig_p)
                    
                    fig_h, ax_h = plt.subplots(figsize=(p_w/2, p_h))
                    sns.histplot(data=combined_df, x=p_type, hue='SOURCE_FILE', kde=True, ax=ax_h)
                    apply_global_legend(ax_h, global_legend_loc, show_legend); st.pyplot(fig_h)
else:
    st.info("Please upload CSV files to begin analysis.")