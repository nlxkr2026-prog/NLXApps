import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import io

# --- [1] 데이터 전처리 로직 (Z-Gap 0번부터 할당 및 대소문자 표준화) ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip().upper() for c in df.columns]
    
    if 'HEIGHT' in df.columns: d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns: d_type, target_cols = "Radius", ['RADIUS']
    elif 'SHIFT_NORM' in df.columns or 'SHIFT_X' in df.columns: 
        d_type = "Shift"
        target_cols = [c for c in ['SHIFT_NORM', 'SHIFT_X', 'SHIFT_Y'] if c in df.columns]
    elif 'X_COORD' in df.columns: d_type, target_cols = "Coordinate", ['X_COORD']
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
        else: df['L_NUM'] = 0
    else: df['L_NUM'] = 0

    df['P_ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else (df['GROUP_ID'] if 'GROUP_ID' in df.columns else None)

    df_clean = df.copy()
    if apply_iqr and d_type != "Coordinate":
        df_clean = df_clean[df_clean['MEAS_VALUE'] != 0]
        if not df_clean.empty:
            q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [Helper] Contour Plot 함수 ---
def plot_heatmap_core(ax, x, y, z, title, x_lab, y_lab, vmin=None, vmax=None, cmap='jet'):
    if len(x) < 5:
        sc = ax.scatter(x, y, c=z, cmap=cmap, s=20)
        if vmin is not None: sc.set_clim(vmin, vmax)
        return sc
    xi = np.linspace(x.min(), x.max(), 100); yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    cp = ax.contourf(xi, yi, zi, levels=15, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.set_title(title); ax.set_xlabel(x_lab); ax.set_ylabel(y_lab)
    return cp

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Expert Professional", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Final Build)")

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 10); p_h = st.sidebar.slider("Plot Height", 3, 15, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Customization")
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")
custom_x_legend = st.sidebar.text_input("X-axis Legend", "X Position (um)")
custom_y_legend = st.sidebar.text_input("Y-axis Legend", "Y Position (um)")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale Range")
v_min = st.sidebar.number_input("Min Limit", value=-10.0); v_max = st.sidebar.number_input("Max Limit", value=10.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE_FILE'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Shift Trend"])

        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1: selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(selected_layer.split(" ")[1])]
            
            active_target = 'MEAS_VALUE'
            if d_type == "Shift":
                avail_cols = [c for c in display_df.columns if c.endswith('_UM')]
                with c2: active_target = st.selectbox("Select Target Shift", avail_cols, index=0)

            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Dot Distribution"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            if chart_type == "Heatmap":
                cp = plot_heatmap_core(ax1, display_df['X_VAL'], display_df['Y_VAL'], display_df[active_target], 
                                      f"{custom_title} ({selected_layer})", custom_x_legend, custom_y_legend,
                                      vmin=v_min if use_custom_scale else None, vmax=v_max if use_custom_scale else None)
                plt.colorbar(cp, ax=ax1, label="Value (um)")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='SOURCE_FILE', y=active_target, ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            elif chart_type == "Dot Distribution":
                sns.stripplot(data=display_df, x='SOURCE_FILE', y=active_target, jitter=True, alpha=0.5, ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            ax1.set_title(f"{custom_title} - {active_target}"); st.pyplot(fig1)

            st.markdown("---")
            st.subheader(f"📊 Summary Statistics ({selected_layer})")
            summary = display_df.groupby('SOURCE_FILE')[active_target].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            summary['3-Sigma'] = summary['std'] * 3
            st.dataframe(summary)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                ax2.set_title(f"Layer Comparison: {custom_title}"); st.pyplot(fig2)
                st.markdown("---")
                st.subheader("📊 Layer Comparison Statistics")
                comp_stats = combined_df.groupby(['SOURCE_FILE', 'L_NUM'])['MEAS_VALUE'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                comp_stats['3-Sigma'] = comp_stats['std'] * 3
                st.dataframe(comp_stats)
                st.download_button("📥 Export Comparison Stats", comp_stats.to_csv(index=False).encode('utf-8'), "layer_comparison_stats.csv")

        with tab3:
            shift_df = combined_df.dropna(subset=['P_ID'])
            if not shift_df.empty and len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment & Shift Trend (Ref: Layer 0)")
                trend_list, heatmap_data_list = [], []
                for src in shift_df['SOURCE_FILE'].unique():
                    src_df = shift_df[shift_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    if not base.empty:
                        for lyr in unique_layers:
                            target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                            merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                            if not merged.empty:
                                merged['DX'] = merged['X_VAL_TGT'] - merged['X_VAL_REF']
                                merged['DY'] = merged['Y_VAL_TGT'] - merged['Y_VAL_REF']
                                merged['MAG'] = np.sqrt(merged['DX']**2 + merged['DY']**2)
                                trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': merged['DX'].mean(), 'Avg_DY': merged['DY'].mean(), 'Avg_Mag': merged['MAG'].mean(), 'Std_Mag': merged['MAG'].std(), 'Count': len(merged)})
                                merged['Layer'] = lyr; merged['Source'] = src; heatmap_data_list.append(merged)

                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X Avg)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Avg)")
                    ax3.axvline(0, color='black', alpha=0.3); ax3.set_yticks(unique_layers)
                    ax3.set_xlabel("Average Shift (um)"); ax3.set_ylabel("Layer Number")
                    ax3.legend(); st.pyplot(fig3)
                    
                    st.markdown("---")
                    st.subheader("📊 Multi-Layer Shift Trend Statistics")
                    st.dataframe(trend_df) # [추가] 트렌드 통계 출력
                    st.download_button("📥 Export Trend CSV", trend_df.to_csv(index=False).encode('utf-8'), "alignment_trend.csv")

                    st.markdown("---")
                    st.subheader("Shift Intensity Map")
                    h_layer = st.selectbox("Select Layer to Heat", unique_layers[1:])
                    h_type = st.radio("Visualize Value", ["Magnitude", "Delta X", "Delta Y"], horizontal=True)
                    h_df_all = pd.concat(heatmap_data_list); h_df = h_df_all[h_df_all['Layer'] == h_layer]
                    h_target = {"Magnitude": "MAG", "Delta X": "DX", "Delta Y": "DY"}[h_type]
                    fig4, ax4 = plt.subplots(figsize=(p_w, p_h))
                    cp_h = plot_heatmap_core(ax4, h_df['X_VAL_REF'], h_df['Y_VAL_REF'], h_df[h_target],
                                           f"Layer {h_layer} {h_type} Map", custom_x_legend, custom_y_legend,
                                           cmap='Reds' if h_type=="Magnitude" else 'RdBu_r',
                                           vmin=v_min if use_custom_scale else None, vmax=v_max if use_custom_scale else None)
                    plt.colorbar(cp_h, label=f"{h_type} (um)"); st.pyplot(fig4)
            else: st.info("Shift 분석을 위해 Pillar 데이터와 Layer 0가 필요합니다.")