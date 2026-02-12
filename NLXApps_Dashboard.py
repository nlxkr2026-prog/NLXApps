import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer Pro", layout="wide")
st.title("🔬 Universal Bump Quality & 3D Interactive Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 및 시각화 설정")

uploaded_files = st.sidebar.file_uploader("분석용 CSV 파일들을 업로드하세요", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
    z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)
    
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ IQR Outlier Filtering")
    use_filter_radius = st.sidebar.checkbox("Filter Radius (IQR)", value=True)
    use_filter_height = st.sidebar.checkbox("Filter Height (IQR)", value=True)
    use_filter_shift = st.sidebar.checkbox("Filter Shift (IQR)", value=False)
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 시각화 옵션")
    layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])
    hist_layout = st.sidebar.selectbox("히스토그램 레이아웃", ["Facet (파일별 분리)", "Overlay (겹쳐보기)"])
    vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 200, 50)

    # --- 3. 로직 함수 ---
    def apply_iqr_filter(series):
        if series.dropna().empty: return series
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return series.mask((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR))

    def preprocess_df(df, scale):
        cols = ['Group_ID', 'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
        for c in df.columns:
            if c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if c != 'Group_ID': df[c] *= scale
        return df

    def get_layer_info(df, gap):
        z_col = next((c for c in ['Bump_Center_Z', 'Z_Coord', 'Intersection_Height'] if c in df.columns), None)
        if z_col and df[z_col].notna().any():
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        else:
            df['Inferred_Layer'] = df['Layer_Number'] if 'Layer_Number' in df.columns else 0
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

    # --- 4. 데이터 로드 및 마스터 매핑 ---
    raw_data = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    st.info("🎯 **Master File**을 선택하거나 **'Independent Analysis'**를 선택하세요.")
    m_options = ["Independent Analysis (No Master)"] + list(raw_data.keys())
    m_key = st.selectbox("Master 파일 선택", m_options)
    
    layer_map, master_coords = None, None
    if m_key != "Independent Analysis (No Master)":
        m_df_p = get_layer_info(raw_data[m_key], z_gap_threshold)
        layer_map = m_df_p[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
        xc_m = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in m_df_p.columns), 'X')
        yc_m = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in m_df_p.columns), 'Y')
        zc_m = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in m_df_p.columns), 'Z')
        master_coords = m_df_p[['Group_ID', xc_m, yc_m, zc_m]].rename(columns={xc_m:'X', yc_m:'Y', zc_m:'Z'}).drop_duplicates()

    processed_list = []
    for name, df in raw_data.items():
        if m_key != "Independent Analysis (No Master)" and name != m_key:
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
        processed_list.append(df)

    if processed_list:
        full_df = pd.concat(processed_list, ignore_index=True)
        st.subheader("📊 Summary Statistics")
        m_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
        st.dataframe(full_df.groupby(['File_Name', 'Inferred_Layer'])[m_list].agg(['mean', 'std', 'count']).round(3), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📏 Group A: 형상 분석", "🎯 Group B: 위치 편차", "🌐 3D View & Highlight"])

        with tab1:
            st.header("Group A: Shape Analysis")
            sel_met_a = st.selectbox("지표 선택 (A)", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in full_df.columns])
            p_df_a = full_df.dropna(subset=[sel_met_a])
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.box(p_df_a, x="File_Name", y=sel_met_a, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True)
            c2.plotly_chart(px.histogram(p_df_a, x=sel_met_a, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True)
            
            # [복구] Heatmap 기능
            st.divider()
            st.subheader("📍 Spatial Heatmap")
            f_map = st.selectbox("지도를 볼 파일 선택 (A)", full_df['File_Name'].unique(), key="map_a")
            m_df_a = full_df[(full_df['File_Name'] == f_map) & (full_df[sel_met_a].notna())]
            xc_a = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in m_df_a.columns), 'X')
            yc_a = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in m_df_a.columns), 'Y')
            st.plotly_chart(px.scatter(m_df_a, x=xc_a, y=yc_a, color=sel_met_a, facet_col="Inferred_Layer", color_continuous_scale="Turbo"), use_container_width=True)

        with tab2:
            st.header("Group B: Alignment Analysis")
            b_metrics = [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
            if b_metrics:
                sel_met_b = st.selectbox("Shift 지표 선택 (B)", b_metrics)
                p_df_b = full_df.dropna(subset=[sel_met_b])
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.box(p_df_b, x="File_Name", y=sel_met_b, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True)
                c2.plotly_chart(px.histogram(p_df_b, x=sel_met_b, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True)
                
                if 'Shift_X' in full_df.columns and 'Shift_Y' in full_df.columns:
                    st.divider()
                    st.subheader("📍 Shift Vector Map")
                    v_file = st.selectbox("화살표 맵 파일 선택", full_df['File_Name'].unique())
                    v_df = full_df[(full_df['File_Name'] == v_file) & full_df['Shift_X'].notna()]
                    if not v_df.empty:
                        xc, yc = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in v_df.columns else ('X_Coord', 'Y_Coord')
                        fig_v = ff.create_quiver(x=v_df[xc], y=v_df[yc], u=v_df['Shift_X']*vector_scale, v=v_df['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                        fig_v.add_trace(go.Scatter(x=v_df[xc], y=v_df[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3)))
                        fig_v.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1))
                        st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.warning("Shift 데이터가 포함된 파일이 없습니다.")

        with tab3:
            st.header("🌐 3D Structural View")
            if m_key != "Independent Analysis (No Master)":
                pivot_df = full_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
                df3 = pivot_df.merge(master_coords, on='Group_ID', how='left')
                x3, y3, z3 = 'X', 'Y', 'Z'
            else:
                t_f = st.selectbox("3D 파일 선택", full_df['File_Name'].unique())
                df3 = full_df[full_df['File_Name'] == t_f].copy()
                x3 = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in df3.columns), 'X')
                y3 = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in df3.columns), 'Y')
                z3 = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in df3.columns), 'Inferred_Layer')

            avail_3d = [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in df3.columns]
            if avail_3d:
                c_3d_met = st.selectbox("색상/하이라이트 지표", avail_3d)
                df3 = df3.dropna(subset=[c_3d_met])
                apply_th = st.checkbox("⚠️ Threshold Highlighting 적용", value=False)
                if apply_th:
                    c1, c2 = st.columns(2)
                    h_th = c1.number_input("High Threshold (Red Above)", value=float(df3[c_3d_met].max()))
                    l_th = c2.number_input("Low Threshold (Yellow Below)", value=float(df3[c_3d_met].min()))
                    df3['Color_Group'] = df3[c_3d_met].apply(lambda v: 'Critical' if v >= h_th else ('Warning' if v <= l_th else 'Normal'))
                    fig3 = px.scatter_3d(df3, x=x3, y=y3, z=z3, color='Color_Group', color_discrete_map={'Critical': 'red', 'Warning': 'yellow', 'Normal': 'lightgray'}, opacity=0.8)
                else:
                    fig3 = px.scatter_3d(df3, x=x3, y=y3, z=z3, color=c_3d_met, color_continuous_scale='Turbo', opacity=0.8)
                fig3.update_layout(scene=dict(aspectmode='data'), height=800)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error("3D로 표현할 수 있는 지표가 없습니다.")
else:
    st.info("👈 CSV 파일들을 업로드하세요.")