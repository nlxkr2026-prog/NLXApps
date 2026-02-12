import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # 3D 시각화용
import io
import os

# --- [1] 데이터 전처리 및 Pitch 계산 로직 ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명 대문자 표준화
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별 및 타겟 설정
    d_type = None
    target_cols = []
    
    if 'HEIGHT' in df.columns: 
        d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns: 
        d_type, target_cols = "Radius", ['RADIUS']
    elif 'SHIFT_NORM' in df.columns or 'SHIFT_X' in df.columns: 
        d_type = "Shift"
        if 'SHIFT_NORM' in df.columns: target_cols.append('SHIFT_NORM')
        if 'SHIFT_X' in df.columns: target_cols.append('SHIFT_X')
        if 'SHIFT_Y' in df.columns: target_cols.append('SHIFT_Y')
    elif 'X_COORD' in df.columns: 
        d_type, target_cols = "Coordinate", ['X_COORD']
    else: return None, None

    # 2. 좌표 및 측정값 설정
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    
    for col in target_cols:
        df[col + '_UM'] = df[col] * scale_factor
    
    df['MEAS_VALUE'] = df[target_cols[0] + '_UM']
    
    # [Pitch 계산 알고리즘 - 행/열 바뀜 방지]
    # X-Pitch: Y좌표가 같은 그룹(행) 내에서 X좌표 차이 계산
    df['Y_GRID'] = df['Y_VAL'].round(1) 
    df = df.sort_values(['SOURCE_FILE' if 'SOURCE_FILE' in df.columns else 'Y_GRID', 'Y_GRID', 'X_VAL'])
    df['X_PITCH'] = df.groupby(['SOURCE_FILE' if 'SOURCE_FILE' in df.columns else 'Y_GRID', 'Y_GRID'])['X_VAL'].diff()

    # Y-Pitch: X좌표가 같은 그룹(열) 내에서 Y좌표 차이 계산
    df['X_GRID'] = df['X_VAL'].round(1)
    df = df.sort_values(['SOURCE_FILE' if 'SOURCE_FILE' in df.columns else 'X_GRID', 'X_GRID', 'Y_VAL'])
    df['Y_PITCH'] = df.groupby(['SOURCE_FILE' if 'SOURCE_FILE' in df.columns else 'X_GRID', 'X_GRID'])['Y_VAL'].diff()

    # Pitch 이상치 제거 (인접하지 않은 Bump 간 계산 방지)
    for p_col in ['X_PITCH', 'Y_PITCH']:
        if not df[p_col].dropna().empty:
            pq1, pq3 = df[p_col].quantile([0.25, 0.75])
            piqr = pq3 - pq1
            df.loc[(df[p_col] > pq3 + 1.5 * piqr), p_col] = np.nan

    # 3. 레이어 번호 설정
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
        else:
            df['L_NUM'] = 0
    else:
        df['L_NUM'] = 0

    df['P_ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else (df['GROUP_ID'] if 'GROUP_ID' in df.columns else None)

    # 5. IQR 필터링
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
        if handles:
            ax.legend(handles=handles, labels=labels, loc=loc, title=None)

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Professional", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

with st.sidebar:
    st.header("📁 Data Config")
    uploaded_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
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
        color_option = st.selectbox("Color Theme", ["Viridis", "Plasma", "Inferno", "Magma", "Jet", "Turbo"])
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
        
        m_col1.metric("Global Avg", f"{avg_val:.3f}")
        m_col2.metric("Global 3-Sigma", f"{(std_val * 3):.3f}")
        m_col3.metric("Max-Min Range", f"{(combined_df['MEAS_VALUE'].max() - combined_df['MEAS_VALUE'].min()):.3f}")
        m_col4.metric("Total Bumps", f"{len(combined_df):,}")

        with st.expander("📄 Detailed Statistics by File"):
            file_stats = combined_df.groupby('SOURCE_FILE')['MEAS_VALUE'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            file_stats['3-Sigma'] = file_stats['std'] * 3
            file_stats.columns = ['File Name', 'Average', 'Std Dev', 'Min', 'Max', 'Count', '3-Sigma']
            st.dataframe(file_stats[['File Name', 'Average', '3-Sigma', 'Min', 'Max', 'Count']].style.format(precision=3), use_container_width=True)
        
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View", "🎯 Pitch Analysis"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1: selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(selected_layer.split(" ")[1])]
            
            if d_type == "Shift":
                with c2: active_target = st.selectbox("Target Data", [c for c in display_df.columns if c.endswith('_UM')], index=0)
            else: active_target = 'MEAS_VALUE'

            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Histogram"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X_VAL'], display_df['Y_VAL'], c=display_df[active_target], cmap='jet', s=15)
                if use_custom_scale: sc.set_clim(v_min, v_max)
                plt.colorbar(sc)
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='SOURCE_FILE', y=active_target, hue='SOURCE_FILE', ax=ax1, palette='Set2')
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
                apply_global_legend(ax1, global_legend_loc, show_legend)
            elif chart_type == "Histogram":
                sns.histplot(data=display_df, x=active_target, hue='SOURCE_FILE', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
                apply_global_legend(ax1, global_legend_loc, show_legend)
            
            ax1.set_title(custom_title); ax1.set_xlabel(x_lbl); ax1.set_ylabel(y_lbl)
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                apply_global_legend(ax2, global_legend_loc, show_legend)
                ax2.set_title(custom_title); st.pyplot(fig2)
            else: st.info("Need more layers.")

        with tab3:
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
                                trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean(), 'Avg_DY': (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()})
                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y)")
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    ax3.set_title(custom_title); apply_global_legend(ax3, global_legend_loc, show_legend)
                    st.pyplot(fig3)

        with tab4:
            st.subheader("Interactive 3D Layer Stack View")
            plot_3d_df = combined_df.copy()
            if use_outlier_filter:
                conditions = [(plot_3d_df['MEAS_VALUE'] < outlier_low), (plot_3d_df['MEAS_VALUE'] > outlier_high)]
                choices = ['Under Limit (Low)', 'Over Limit (High)']
                plot_3d_df['Status'] = np.select(conditions, choices, default='Normal')
                fig_3d = px.scatter_3d(
                    plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM',
                    color='Status',
                    color_discrete_map={'Under Limit (Low)': 'yellow', 'Over Limit (High)': 'red', 'Normal': 'blue'},
                    opacity=0.6, labels={'X_VAL': 'X', 'Y_VAL': 'Y', 'L_NUM': 'Layer'},
                    title=f"3D Outlier Highlight: {custom_title}"
                )
            else:
                fig_3d = px.scatter_3d(
                    plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM',
                    color='MEAS_VALUE', size_max=5, opacity=0.7,
                    color_continuous_scale=color_option.lower(),
                    labels={'X_VAL': 'X', 'Y_VAL': 'Y', 'L_NUM': 'Layer'},
                    title=f"3D Distribution: {custom_title}"
                )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab5:
            st.subheader("🎯 Pitch Analysis (X & Y Distribution)")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**X-Pitch Analysis**")
                fig_px, ax_px = plt.subplots(figsize=(p_w/2, p_h))
                sns.boxplot(data=combined_df, x='SOURCE_FILE', y='X_PITCH', hue='SOURCE_FILE', ax=ax_px, palette='Blues')
                ax_px.set_title("X-Pitch Boxplot"); apply_global_legend(ax_px, global_legend_loc, show_legend)
                st.pyplot(fig_px)
                
                fig_hx, ax_hx = plt.subplots(figsize=(p_w/2, p_h))
                sns.histplot(data=combined_df, x='X_PITCH', hue='SOURCE_FILE', kde=True, ax=ax_hx)
                ax_hx.set_title("X-Pitch Histogram"); apply_global_legend(ax_hx, global_legend_loc, show_legend)
                st.pyplot(fig_hx)

            with col_p2:
                st.markdown("**Y-Pitch Analysis**")
                fig_py, ax_py = plt.subplots(figsize=(p_w/2, p_h))
                sns.boxplot(data=combined_df, x='SOURCE_FILE', y='Y_PITCH', hue='SOURCE_FILE', ax=ax_py, palette='Reds')
                ax_py.set_title("Y-Pitch Boxplot"); apply_global_legend(ax_py, global_legend_loc, show_legend)
                st.pyplot(fig_py)

                fig_hy, ax_hy = plt.subplots(figsize=(p_w/2, p_h))
                sns.histplot(data=combined_df, x='Y_PITCH', hue='SOURCE_FILE', kde=True, ax=ax_hy)
                ax_hy.set_title("Y-Pitch Histogram"); apply_global_legend(ax_hy, global_legend_loc, show_legend)
                st.pyplot(fig_hy)

else:
    st.info("Upload CSV files to start.")