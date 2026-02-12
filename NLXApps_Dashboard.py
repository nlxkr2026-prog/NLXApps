import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # 3D 시각화용
import io
import os

# --- [1] 데이터 전처리 로직 ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip().upper() for c in df.columns]
    
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

    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    
    for col in target_cols:
        df[col + '_UM'] = df[col] * scale_factor
    
    df['MEAS_VALUE'] = df[target_cols[0] + '_UM']
    
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

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_multiplier = st.sidebar.slider("Plot Scale (Multiplier)", 0.5, 3.0, 1.0, 0.1)
p_w = 10 * p_multiplier
p_h = 6 * p_multiplier

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Legend & Text Settings")
custom_title = st.sidebar.text_input("Graph Title", "Alignment Analysis")
custom_x_legend = st.sidebar.text_input("X-axis Legend Name", "X Position (um)")
custom_y_legend = st.sidebar.text_input("Y-axis Legend Name", "Y Position (um)")

show_legend = st.sidebar.checkbox("Show Legend", value=True)
global_legend_loc = st.sidebar.selectbox(
    "Global Legend Location", 
    options=["best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"],
    index=1,
    disabled=not show_legend
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale Range", value=False)
v_min = st.sidebar.number_input("Min Limit", value=-10.0)
v_max = st.sidebar.number_input("Max Limit", value=10.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🧊 3D View Settings")
color_option = st.sidebar.selectbox("3D Color Scale", ["Viridis", "Plasma", "Inferno", "Magma", "Jet", "Turbo"])
st.sidebar.write("**Outlier Highlight (Red)**")
use_outlier_filter = st.sidebar.checkbox("Highlight Values Outside Range")
outlier_min = st.sidebar.number_input("Outlier Lower Bound", value=-5.0)
outlier_max = st.sidebar.number_input("Outlier Upper Bound", value=5.0)

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
        avg_val = combined_df['MEAS_VALUE'].mean()
        std_val = combined_df['MEAS_VALUE'].std()
        count_val = len(combined_df)
        
        m_col1.metric("Global Average", f"{avg_val:.3f} um")
        m_col2.metric("Global 3-Sigma", f"{(std_val * 3):.3f} um")
        m_col3.metric("Global Max-Min", f"{(combined_df['MEAS_VALUE'].max() - combined_df['MEAS_VALUE'].min()):.3f} um")
        m_col4.metric("Total Bumps", f"{count_val:,}")

        with st.expander("📄 View File-wise Detailed Statistics", expanded=True):
            file_stats = combined_df.groupby('SOURCE_FILE')['MEAS_VALUE'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            file_stats['3-Sigma'] = file_stats['std'] * 3
            file_stats.columns = ['File Name', 'Average (um)', 'Std Dev', 'Min', 'Max', 'Count', '3-Sigma (um)']
            file_stats = file_stats[['File Name', 'Average (um)', '3-Sigma (um)', 'Min', 'Max', 'Count']]
            st.dataframe(file_stats.style.format(subset=['Average (um)', '3-Sigma (um)', 'Min', 'Max'], formatter="{:.3f}"), use_container_width=True)
        
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View"
        ])

        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(selected_layer.split(" ")[1])]
            active_target = 'MEAS_VALUE'
            if d_type == "Shift":
                avail_cols = [c for c in display_df.columns if c.endswith('_UM')]
                with c2: active_target = st.selectbox("Select Target Data", avail_cols, index=0)

            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Histogram"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X_VAL'], display_df['Y_VAL'], c=display_df[active_target], cmap='jet', s=15)
                if use_custom_scale: sc.set_clim(v_min, v_max)
                plt.colorbar(sc, label=f"Value (um)")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='SOURCE_FILE', y=active_target, hue='SOURCE_FILE', ax=ax1, palette='Set2')
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
                apply_global_legend(ax1, global_legend_loc, show_legend)
            elif chart_type == "Histogram":
                sns.histplot(data=display_df, x=active_target, hue='SOURCE_FILE', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
                apply_global_legend(ax1, global_legend_loc, show_legend)
            
            ax1.set_title(custom_title)
            ax1.set_xlabel(custom_x_legend); ax1.set_ylabel(custom_y_legend)
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                apply_global_legend(ax2, global_legend_loc, show_legend)
                ax2.set_title(custom_title)
                ax2.set_xlabel("Layer Number"); ax2.set_ylabel(f"{d_type} Value")
                st.pyplot(fig2)
            else: st.info("Need more than 1 layer for comparison.")

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
                    
                    # [추가] Shift Trend X축 스케일 적용
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    
                    ax3.set_title(custom_title)
                    ax3.set_xlabel("Average Shift (um)"); ax3.set_ylabel("Layer Number")
                    apply_global_legend(ax3, global_legend_loc, show_legend)
                    st.pyplot(fig3)

        with tab4:
            st.subheader("Interactive 3D Layer Stack View")
            plot_3d_df = combined_df.copy()
            if use_outlier_filter:
                plot_3d_df['Status'] = np.where(
                    (plot_3d_df['MEAS_VALUE'] < outlier_min) | (plot_3d_df['MEAS_VALUE'] > outlier_max),
                    'Outlier', 'Normal'
                )
                fig_3d = px.scatter_3d(
                    plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM',
                    color='Status',
                    color_discrete_map={'Outlier': 'red', 'Normal': 'blue'},
                    opacity=0.6,
                    labels={'X_VAL': 'X (um)', 'Y_VAL': 'Y (um)', 'L_NUM': 'Layer'},
                    title=f"3D View with Outliers: {custom_title}"
                )
            else:
                fig_3d = px.scatter_3d(
                    plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM',
                    color='MEAS_VALUE', size_max=5, opacity=0.7,
                    color_continuous_scale=color_option.lower(),
                    labels={'X_VAL': 'X (um)', 'Y_VAL': 'Y (um)', 'L_NUM': 'Layer'},
                    title=f"3D Distribution: {custom_title}"
                )
            
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("Please upload CSV files to start analysis.")