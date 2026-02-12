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
    # 컬럼명 표준화 (공백 제거 및 대문자화)
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

    # 3. 레이어 번호 설정 (Pitch 계산 전 필수)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
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

    # 4. [핵심] 개선된 Pitch 알고리즘 (GroupID 연속성 및 좌표 그룹화 반영)
    group_base = ['SOURCE_FILE', 'L_NUM'] if 'SOURCE_FILE' in df.columns else ['L_NUM']
    df['P_ID'] = df['GROUP_ID'] if 'GROUP_ID' in df.columns else df.index

    # [X-Pitch] 동일 레이어 내에서 Y좌표가 거의 같은 범프들을 한 줄로 인식
    # 소수점 셋째 자리까지 일치하면 동일 행으로 간주하여 정밀도 향상
    df['Y_ROUND'] = df['Y_VAL'].round(3)
    df = df.sort_values(group_base + ['Y_ROUND', 'X_VAL'])
    
    # GroupID가 1씩 증가할 때만 인접 범프로 간주하여 X 차이 계산
    df['ID_STEP'] = df.groupby(group_base + ['Y_ROUND'])['P_ID'].diff()
    df['X_PITCH'] = np.where(df['ID_STEP'] == 1, df.groupby(group_base + ['Y_ROUND'])['X_VAL'].diff().abs(), np.nan)

    # [Y-Pitch] X좌표가 거의 같은 범프들을 한 열로 인식
    df['X_ROUND'] = df['X_VAL'].round(3)
    df = df.sort_values(group_base + ['X_ROUND', 'Y_VAL'])
    # 열 방향은 ID가 건너뛰는 경우가 많으므로 좌표 차이 기반으로 인접성 판단 (예: 50um 이내)
    df['Y_PITCH_RAW'] = df.groupby(group_base + ['X_ROUND'])['Y_VAL'].diff().abs()
    df['Y_PITCH'] = np.where(df['Y_PITCH_RAW'] < 50, df['Y_PITCH_RAW'], np.nan) # 50um 이상은 행 바뀜으로 간주

    # 5. IQR 필터링 (최종 결과값 정제)
    df_clean = df.copy()
    if apply_iqr and d_type != "Coordinate":
        df_clean = df_clean[df_clean['MEAS_VALUE'] != 0]
        if not df_clean.empty:
            q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

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
    # 버튼 없이 입력만 받는 형식
    scale = st.number_input("Multiplier (Scale Factor)", value=1.0, format="%.4f")
    use_iqr = st.checkbox("Apply IQR Filter", value=True)

    with st.expander("🎨 Plot Settings", expanded=True):
        p_w = st.slider("Plot Width", 5, 25, 12)
        p_h = st.slider("Plot Height", 3, 15, 6)
        custom_title = st.text_input("Graph Title", "Alignment Analysis")
        x_lbl = st.text_input("X Axis Label", "X Position (um)")
        y_lbl = st.text_input("Y Axis Label", "Y Position (um)")

    with st.expander("📏 Legend & Scale Control", expanded=False):
        show_legend = st.checkbox("Show Legend", value=True)
        global_legend_loc = st.selectbox("Legend Loc", 
            options=["best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"], 
            index=1, disabled=not show_legend)
        st.markdown("---")
        use_custom_scale = st.checkbox("Manual Axis Range", value=False)
        v_min = st.number_input("Min Limit", value=-10.0)
        v_max = st.number_input("Max Limit", value=10.0)

    with st.expander("🧊 3D & Outlier Settings", expanded=False):
        color_option = st.selectbox("Color Theme", ["Viridis", "Plasma", "Jet", "Turbo"])
        st.markdown("---")
        use_outlier_filter = st.checkbox("Highlight Outliers")
        outlier_low = st.number_input("Lower Bound (Yellow)", value=-5.0)
        outlier_high = st.number_input("Upper Bound (Red)", value=5.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE_FILE'] = os.path.splitext(file.name)[0]
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())

        st.markdown("### 📋 Quick Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        avg_val, std_val = combined_df['MEAS_VALUE'].mean(), combined_df['MEAS_VALUE'].std()
        m_col1.metric("Global Average", f"{avg_val:.3f} um")
        m_col2.metric("Global 3-Sigma", f"{(std_val * 3):.3f} um")
        m_col3.metric("Max-Min Range", f"{(combined_df['MEAS_VALUE'].max() - combined_df['MEAS_VALUE'].min()):.3f} um")
        m_col4.metric("Total Bumps", f"{len(combined_df):,}")

        with st.expander("📄 View File-wise Detailed Statistics"):
            file_stats = combined_df.groupby('SOURCE_FILE')['MEAS_VALUE'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            file_stats['3-Sigma'] = file_stats['std'] * 3
            st.dataframe(file_stats.style.format(precision=3), use_container_width=True)
        
        st.markdown("---")
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

        with tabs[1]:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                apply_global_legend(ax2, global_legend_loc, show_legend)
                ax2.set_title("Layer Comparison"); st.pyplot(fig2)
            else: st.info("Requires more than one layer for comparison.")

        with tabs[2]:
            shift_df = combined_df.dropna(subset=['P_ID'])
            if not shift_df.empty and len(unique_layers) > 1:
                trend_list = []
                for src in shift_df['SOURCE_FILE'].unique():
                    src_df = shift_df[shift_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    if not base.empty:
                        for lyr in unique_layers:
                            target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                            merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                            if not merged.empty:
                                trend_list.append({'Source': src, 'Layer': lyr, 
                                                   'Avg_DX': (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean(), 
                                                   'Avg_DY': (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()})
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y)")
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    ax3.set_title("Shift Trend"); apply_global_legend(ax3, global_legend_loc, show_legend)
                    st.pyplot(fig3)

        with tabs[3]:
            st.subheader("Interactive 3D Layer Stack View")
            plot_3d_df = combined_df.copy()
            if use_outlier_filter:
                conditions = [(plot_3d_df['MEAS_VALUE'] < outlier_low), (plot_3d_df['MEAS_VALUE'] > outlier_high)]
                choices = ['Under Limit (Low)', 'Over Limit (High)']
                plot_3d_df['Status'] = np.select(conditions, choices, default='Normal')
                fig_3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='Status',
                                     color_discrete_map={'Under Limit (Low)': 'yellow', 'Over Limit (High)': 'red', 'Normal': 'blue'},
                                     opacity=0.6, title=custom_title)
            else:
                fig_3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='MEAS_VALUE', color_continuous_scale=color_option.lower())
            fig_3d.update_layout(height=700); st.plotly_chart(fig_3d, use_container_width=True)

        with tabs[4]:
            st.subheader("🎯 Pitch Analysis (X & Y Distribution)")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**X-Pitch Analysis (Between Columns)**")
                fig_px, ax_px = plt.subplots(figsize=(p_w/2, p_h))
                sns.boxplot(data=combined_df, x='SOURCE_FILE', y='X_PITCH', hue='SOURCE_FILE', ax=ax_px, palette='Blues')
                apply_global_legend(ax_px, global_legend_loc, show_legend); st.pyplot(fig_px)
                
                fig_hx, ax_hx = plt.subplots(figsize=(p_w/2, p_h))
                sns.histplot(data=combined_df, x='X_PITCH', hue='SOURCE_FILE', kde=True, ax=ax_hx)
                apply_global_legend(ax_hx, global_legend_loc, show_legend); st.pyplot(fig_hx)

            with col_p2:
                st.markdown("**Y-Pitch Analysis (Between Rows)**")
                fig_py, ax_py = plt.subplots(figsize=(p_w/2, p_h))
                sns.boxplot(data=combined_df, x='SOURCE_FILE', y='Y_PITCH', hue='SOURCE_FILE', ax=ax_py, palette='Reds')
                apply_global_legend(ax_py, global_legend_loc, show_legend); st.pyplot(fig_py)

                fig_hy, ax_hy = plt.subplots(figsize=(p_w/2, p_h))
                sns.histplot(data=combined_df, x='Y_PITCH', hue='SOURCE_FILE', kde=True, ax=ax_hy)
                apply_global_legend(ax_hy, global_legend_loc, show_legend); st.pyplot(fig_hy)
else:
    st.info("Please upload CSV files to begin analysis.")