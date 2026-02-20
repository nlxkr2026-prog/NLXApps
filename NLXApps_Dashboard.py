import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO

# --- 1. Page Configuration ---
st.set_page_config(page_title="Bump Master Analyzer Pro", layout="wide")
st.title("🔬 Universal Bump Quality Analyzer (v.Final_Fixed)")

# --- 2. Sidebar Settings & File Upload ---
st.sidebar.header("⚙️ Analysis & Visualization Settings")

# Session state key for resetting the file uploader (Remove All feature)
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if st.sidebar.button("🗑️ Remove All Files"):
    st.session_state["file_uploader_key"] += 1
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload all CSV files", 
    type=['csv'], 
    accept_multiple_files=True,
    key=st.session_state["file_uploader_key"]
)

if uploaded_files:
    scale_factor = st.sidebar.selectbox("Scale Factor", [1, 1000], index=0, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
    z_gap_threshold = st.sidebar.slider("Z-Gap Layering Threshold (um)", 10, 500, 50)
    
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ IQR Outlier Filtering")
    use_filter_radius = st.sidebar.checkbox("Filter Radius (IQR)", value=False)
    use_filter_height = st.sidebar.checkbox("Filter Height (IQR)", value=False)
    use_filter_shift = st.sidebar.checkbox("Filter Shift (IQR)", value=False)
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 Visualization Options")
    layer_view_mode = st.sidebar.radio("Layer Display Mode", ["Layer All", "Split by Layer"])
    hist_layout = st.sidebar.selectbox("Histogram Layout", ["Facet (Split by File)", "Overlay"])
    vector_scale = st.sidebar.slider("Vector Scale (Arrow Size)", 1, 200, 5)

    # --- 3. Logic Functions ---
    def load_and_adapt(file_obj):
        df = pd.read_csv(file_obj)
        
        # Check for Multilayer Align data (presence of Pillar_Number)
        if 'Pillar_Number' in df.columns:
            # Standardize coordinate column names
            rename_map = {
                'X_Coord': 'Bump_Center_X',
                'Y_Coord': 'Bump_Center_Y',
                'Z_Coord': 'Bump_Center_Z'
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Auto-calculate Shift data based on Pillar_Number
            if 'Layer_Number' in df.columns:
                ref_df = df[df['Layer_Number'] == 0][['Pillar_Number', 'Bump_Center_X', 'Bump_Center_Y']]
                ref_df = ref_df.rename(columns={'Bump_Center_X': 'Ref_X', 'Bump_Center_Y': 'Ref_Y'})
                
                df = df.merge(ref_df, on='Pillar_Number', how='left')
                df['Shift_X'] = df['Bump_Center_X'] - df['Ref_X']
                df['Shift_Y'] = df['Bump_Center_Y'] - df['Ref_Y']
                df['Shift_Norm'] = np.sqrt(df['Shift_X']**2 + df['Shift_Y']**2)
                df = df.drop(columns=['Ref_X', 'Ref_Y'])
            
            # Create dummy Height if absent
            if 'Height' not in df.columns and 'Bump_Center_Z' in df.columns:
                df['Height'] = df['Bump_Center_Z']
                
        return df

    def apply_iqr_filter(series):
        if series.dropna().empty: return series
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return series.mask((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR))

    def preprocess_df(df, scale):
        cols_to_scale = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
        for c in df.columns:
            if c in cols_to_scale:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] *= scale
            if c == 'Group_ID':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def get_layer_info(df, gap):
        if 'Layer_Number' in df.columns:
            df['Inferred_Layer'] = df['Layer_Number']
            return df
            
        z_col = next((c for c in ['Bump_Center_Z', 'Z_Coord', 'Intersection_Height'] if c in df.columns), None)
        if z_col and df[z_col].notna().any():
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        else:
            df['Inferred_Layer'] = 0
        return df

    def calculate_xy_pitch(df):
        if 'Height' not in df.columns: return df
        x_c = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in df.columns), None)
        y_c = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in df.columns), None)
        if not x_c or not y_c: return df
        res = []
        for l in df['Inferred_Layer'].unique():
            ldf = df[df['Inferred_Layer'] == l].copy()
            if len(ldf) < 2: 
                res.append(ldf); continue
            ldf['Y_G'] = ldf[y_c].round(0)
            ldf = ldf.sort_values(['Y_G', x_c])
            ldf['Pitch_X'] = ldf.groupby('Y_G')[x_c].diff().abs()
            ldf['X_G'] = ldf[x_c].round(0)
            ldf = ldf.sort_values(['X_G', y_c])
            ldf['Pitch_Y'] = ldf.groupby('X_G')[y_c].diff().abs()
            ldf['Pitch_X'], ldf['Pitch_Y'] = apply_iqr_filter(ldf['Pitch_X']), apply_iqr_filter(ldf['Pitch_Y'])
            res.append(ldf)
        return pd.concat(res) if res else df

    # --- 4. Main Data Processing Engine ---
    raw_data_dict = {f.name: preprocess_df(load_and_adapt(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Select a Master File**. All data will be merged based on this file's layer structure.")
    m_options = ["Independent Analysis"] + list(raw_data_dict.keys())
    m_key = st.selectbox("Select Master File", m_options)
    
    layer_map, master_coords = None, None
    if m_key != "Independent Analysis":
        m_df_base = get_layer_info(raw_data_dict[m_key], z_gap_threshold)
        layer_map = m_df_base[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
        xc_m = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in m_df_base.columns), 'X')
        yc_m = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in m_df_base.columns), 'Y')
        zc_m = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in m_df_base.columns), 'Z')
        master_coords = m_df_base[['Group_ID', xc_m, yc_m, zc_m]].rename(columns={xc_m:'X', yc_m:'Y', zc_m:'Z'}).drop_duplicates()

    final_processed_list = []
    for name, df in raw_data_dict.items():
        if m_key != "Independent Analysis" and name != m_key:
            if 'Group_ID' in df.columns:
                df = df.merge(layer_map, on='Group_ID', how='inner')
            else: continue
        else:
            df = get_layer_info(df, z_gap_threshold)
        
        df = calculate_xy_pitch(df)
        if use_filter_height and 'Height' in df.columns: df['Height'] = apply_iqr_filter(df['Height'])
        if use_filter_radius and 'Radius' in df.columns: df['Radius'] = apply_iqr_filter(df['Radius'])
        if use_filter_shift:
            for sc in ['Shift_X', 'Shift_Y', 'Shift_Norm']:
                if sc in df.columns: df[sc] = apply_iqr_filter(df[sc])
        df['File_Name'] = name
        final_processed_list.append(df)

    if final_processed_list:
        total_df = pd.concat(final_processed_list, ignore_index=True)
        plot_config = {'editable': True, 'displaylogo': False}

        # Data Separation: Single Layer vs Multilayer Data
        has_multi = 'Pillar_Number' in total_df.columns
        if has_multi:
            multi_df = total_df[total_df['Pillar_Number'].notna()].copy()
            single_df = total_df[total_df['Pillar_Number'].isna()].copy()
        else:
            multi_df = pd.DataFrame()
            single_df = total_df.copy()

        # Convert 0 to NaN in single_df for relevant columns to avoid skewed statistics for empty layers
        zero_filter_cols = ['Radius', 'Height', 'Shift_Norm']
        for c in zero_filter_cols:
            if c in single_df.columns:
                single_df[c] = single_df[c].replace(0, np.nan)

        # --- 5. Data Export & Summary View ---
        st.subheader("📁 Data Export & Summary Preview")
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            # [A] Single Layer Summary
            if not single_df.empty:
                st.markdown("#### 🟢 Bump Shape & Shift Summary")
                s_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in single_df.columns]
                if s_list:
                    s_summary_layer = single_df.groupby(['File_Name', 'Inferred_Layer'])[s_list].agg(['mean', 'std', 'count']).round(3)
                    
                    # Remove layers where count is 0
                    counts_layer = s_summary_layer.xs('count', axis=1, level=1)
                    s_summary_layer = s_summary_layer[counts_layer.sum(axis=1) > 0]
                    
                    s_summary_total = single_df.groupby(['File_Name'])[s_list].agg(['mean', 'std', 'count']).round(3)
                    
                    st.dataframe(s_summary_layer, use_container_width=True)
                    s_summary_layer.to_excel(writer, sheet_name='Bump_Layer_Stats')
                    s_summary_total.to_excel(writer, sheet_name='Bump_Total_Stats')
                    single_df.to_excel(writer, sheet_name='Bump_Raw_Data', index=False)

            # [B] Multilayer Summary
            if not multi_df.empty:
                st.markdown("#### 🏢 Multilayer Alignment Summary")
                st.caption("※ Note: In the table below, 'Height' refers to the Z-coordinate (elevation) of the layer, and 'Shift' refers to the relative misalignment from Layer 0.")
                
                m_df_summary = multi_df.copy()
                rename_dict = {
                    'Height': 'Layer_Z_Height(um)', 
                    'Shift_X': 'Align_Shift_X(um)', 
                    'Shift_Y': 'Align_Shift_Y(um)', 
                    'Shift_Norm': 'Align_Shift_Norm(um)'
                }
                m_df_summary = m_df_summary.rename(columns={k: v for k, v in rename_dict.items() if k in m_df_summary.columns})
                
                m_list = [v for v in rename_dict.values() if v in m_df_summary.columns]
                if m_list:
                    m_summary_layer = m_df_summary.groupby(['File_Name', 'Inferred_Layer'])[m_list].agg(['mean', 'std', 'count']).round(3)
                    
                    # Remove layers where count is 0
                    counts_m_layer = m_summary_layer.xs('count', axis=1, level=1)
                    m_summary_layer = m_summary_layer[counts_m_layer.sum(axis=1) > 0]

                    st.dataframe(m_summary_layer, use_container_width=True)
                    m_summary_layer.to_excel(writer, sheet_name='Align_Layer_Stats')
                    multi_df.to_excel(writer, sheet_name='Align_Raw_Data', index=False)

        st.download_button(label="📥 Download Excel Report", data=output.getvalue(), file_name="Bump_Quality_Report.xlsx", mime="application/vnd.ms-excel")
        st.divider()

        # --- Tab Layout ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📏 Group A (Shape)", "🎯 Group B (Shift)", "🌐 3D View", "🏢 Multilayer Align", "🔍 Master Bump Tracker"])

        with tab1:
            st.header("Group A: Shape Analysis")
            if not single_df.empty:
                sel_a = st.selectbox("Select Metric (A)", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in single_df.columns])
                pdf_a = single_df.dropna(subset=[sel_a])
                if not pdf_a.empty:
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(px.box(pdf_a, x="File_Name", y=sel_a, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False, title=f"{sel_a} Boxplot"), use_container_width=True, config=plot_config)
                    c2.plotly_chart(px.histogram(pdf_a, x=sel_a, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, title=f"{sel_a} Distribution"), use_container_width=True, config=plot_config)
                    
                    st.divider()
                    st.subheader(f"📍 Spatial Heatmap: {sel_a}")
                    f_m_a = st.selectbox("Select File for Heatmap", pdf_a['File_Name'].unique(), key="ma_f")
                    m_a_df = pdf_a[pdf_a['File_Name'] == f_m_a]
                    if not m_a_df.empty:
                        xc_h, yc_h = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in m_a_df.columns else ('X_Coord', 'Y_Coord')
                        fig_heat = px.scatter(
                            m_a_df, x=xc_h, y=yc_h, color=sel_a, 
                            facet_col="Inferred_Layer", color_continuous_scale="Turbo",
                            hover_data=['Group_ID']
                        )
                        fig_heat.update_yaxes(scaleanchor="x", scaleratio=1)
                        st.plotly_chart(fig_heat, use_container_width=True, config=plot_config)
            else:
                st.info("This tab is for single layer (Shape) analysis. Please check the '🏢 Multilayer Align' tab for multilayer data.")

        with tab2:
            st.header("Group B: Shift Analysis")
            if not single_df.empty:
                b_mets = [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in single_df.columns]
                if b_mets:
                    sel_b = st.selectbox("Select Shift Metric", b_mets)
                    pdf_b = single_df.dropna(subset=[sel_b])
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(px.box(pdf_b, x="File_Name", y=sel_b, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True, config=plot_config)
                    c2.plotly_chart(px.histogram(pdf_b, x=sel_b, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True, config=plot_config)
                    
                    st.divider()
                    st.subheader("📍 Shift Vector Map")
                    v_f = st.selectbox("Select File for Vector Map", pdf_b['File_Name'].unique(), key="v_f_b")
                    v_d = pdf_b[(pdf_b['File_Name'] == v_f) & pdf_b['Shift_X'].notna()]
                    if not v_d.empty:
                        xc, yc = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in v_d.columns else ('X_Coord', 'Y_Coord')
                        fig_v = ff.create_quiver(x=v_d[xc], y=v_d[yc], u=v_d['Shift_X']*vector_scale, v=v_d['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                        fig_v.add_trace(go.Scatter(
                            x=v_d[xc], y=v_d[yc], mode='markers', 
                            marker=dict(size=3, color='blue', opacity=0.3), name='Bump Center',
                            text=v_d['Group_ID'],
                            hovertemplate='Group_ID: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>'
                        ))
                        fig_v.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
                        st.plotly_chart(fig_v, use_container_width=True, config=plot_config)
            else:
                st.info("This tab is for single layer (Shift) analysis. Please check the '🏢 Multilayer Align' tab for multilayer data.")

        with tab3:
            st.header("🌐 3D Structural View")
            if not single_df.empty:
                if m_key != "Independent Analysis":
                    pivot_df = single_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
                    df3 = pivot_df.merge(master_coords, on='Group_ID', how='left')
                    x3, y3, z3 = 'X', 'Y', 'Z'
                else:
                    t_3 = st.selectbox("Select File for 3D View", single_df['File_Name'].unique())
                    df3 = single_df[single_df['File_Name'] == t_3].copy()
                    x3 = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in df3.columns), 'X')
                    y3 = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in df3.columns), 'Y')
                    z3 = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in df3.columns), 'Inferred_Layer')

                avail_3d = [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in df3.columns]
                if avail_3d:
                    c_3 = st.selectbox("Select 3D Color Metric", avail_3d)
                    df3 = df3.dropna(subset=[c_3])
                    
                    apply_th = st.checkbox("⚠️ Threshold Highlight Mode", value=False)
                    if apply_th:
                        cx, cy = st.columns(2)
                        hth = cx.number_input("High Threshold (Red Above)", value=float(df3[c_3].max()))
                        lth = cy.number_input("Low Threshold (Yellow Below)", value=float(df3[c_3].min()))
                        
                        def get_status(v):
                            if v >= hth: return 'Critical (Red)'
                            elif v <= lth: return 'Warning (Yellow)'
                            else: return 'Normal'
                        
                        df3['Color_Status'] = df3[c_3].apply(get_status)
                        
                        fig3 = px.scatter_3d(
                            df3, x=x3, y=y3, z=z3, 
                            color='Color_Status',
                            color_discrete_map={'Critical (Red)': 'red', 'Warning (Yellow)': 'yellow', 'Normal': 'lightgray'},
                            opacity=0.8,
                            hover_data=['Group_ID', c_3]
                        )
                    else:
                        fig3 = px.scatter_3d(
                            df3, x=x3, y=y3, z=z3, color=c_3, color_continuous_scale='Turbo', opacity=0.8,
                            hover_data=['Group_ID']
                        )
                    
                    fig3.update_layout(scene=dict(aspectmode='data'), height=800, title=f"3D Map: {c_3} (Editable)")
                    st.plotly_chart(fig3, use_container_width=True, config=plot_config)
            else:
                 st.info("This tab is for single layer 3D analysis. Please check the '🏢 Multilayer Align' tab for multilayer data.")

        with tab4:
            st.header("🏢 Multilayer Alignment Analysis")
            if not multi_df.empty:
                st.markdown("Analyze how each Pillar is vertically aligned across different layers.")

                st.subheader("📈 Vertical Shift Trend")
                trend_df = multi_df.groupby(['File_Name', 'Inferred_Layer'])[['Shift_X', 'Shift_Y', 'Shift_Norm']].mean().reset_index()
                fig_trend = go.Figure()
                for fname in trend_df['File_Name'].unique():
                    f_df = trend_df[trend_df['File_Name'] == fname]
                    fig_trend.add_trace(go.Scatter(x=f_df['Shift_X'], y=f_df['Inferred_Layer'], mode='lines+markers', name=f"{fname} (X)"))
                    fig_trend.add_trace(go.Scatter(x=f_df['Shift_Y'], y=f_df['Inferred_Layer'], mode='lines+markers', name=f"{fname} (Y)"))
                fig_trend.update_layout(xaxis_title="Average Shift (um)", yaxis_title="Layer Number", height=600, title="Average Misalignment by Layer")
                st.plotly_chart(fig_trend, use_container_width=True, config=plot_config)

                st.divider()
                st.subheader("📍 Layer Alignment Vector Map")
                avail_layers = sorted(multi_df['Inferred_Layer'].unique())
                if len(avail_layers) > 1:
                    l_sel = st.selectbox("Select Target Layer", avail_layers[1:])
                    l_df = multi_df[multi_df['Inferred_Layer'] == l_sel]
                    if not l_df.empty:
                        xc, yc = 'Bump_Center_X', 'Bump_Center_Y'
                        fig_v_m = ff.create_quiver(x=l_df[xc], y=l_df[yc], u=l_df['Shift_X']*vector_scale, v=l_df['Shift_Y']*vector_scale,
                                                   scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                        fig_v_m.add_trace(go.Scatter(
                            x=l_df[xc], y=l_df[yc], mode='markers', 
                            marker=dict(size=3, color='blue', opacity=0.3), name=f'Layer {l_sel} Center',
                            text=l_df['Group_ID'],
                            hovertemplate='Group_ID: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>'
                        ))
                        fig_v_m.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1), title=f"Layer {l_sel} Shift Vector Map (Scale: x{vector_scale})")
                        st.plotly_chart(fig_v_m, use_container_width=True, config=plot_config)
            else:
                st.info("No multilayer (Pillar) data uploaded.")

        with tab5:
            st.header("🔍 Master Bump Tracker")
            
            if m_key == "Independent Analysis":
                st.warning("🚨 This tab is activated **only when a 'Master File' is selected** for merging data. Please select a reference file from the sidebar.")
            else:
                st.markdown("Track the metrics of a specific bump (`Group_ID`) across all uploaded files and layers.")
                
                valid_groups = sorted(total_df['Group_ID'].dropna().unique())
                
                if valid_groups:
                    min_id = int(min(valid_groups))
                    max_id = int(max(valid_groups))
                    target_id = st.number_input(f"Enter Bump ID (Group_ID) [Available Range: {min_id} ~ {max_id}]", min_value=min_id, max_value=max_id, value=min_id, step=1)
                    
                    if target_id in valid_groups:
                        bump_df = total_df[total_df['Group_ID'] == target_id].copy()
                        
                        st.divider()
                        st.subheader(f"🏷️ Profile of Bump [ID: {int(target_id)}]")
                        
                        view_cols = [c for c in ['Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in bump_df.columns]
                        
                        if view_cols:
                            pivot_view = bump_df.set_index(['Inferred_Layer', 'File_Name'])[view_cols].dropna(how='all', axis=1)
                            st.dataframe(pivot_view, use_container_width=True)
                        else:
                            st.info("No valid metrics found for this Bump ID.")
                    else:
                        st.error(f"❌ Bump ID '{target_id}' does not exist in the uploaded data. Please enter a valid ID.")
                else:
                    st.error("No integrated Group_ID data available.")

else:
    st.info("Please upload CSV files to activate the dashboard.")