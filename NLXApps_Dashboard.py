import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer Pro", layout="wide")
st.title("🔬 Universal Bump Quality & 3D Vector Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader("분석할 CSV 파일들을 업로드하세요", type=['csv'], accept_multiple_files=True)

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

    # --- 3. 핵심 로직 함수 ---

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
        # 좌표 컬럼 유연한 대응
        z_col = 'Bump_Center_Z' if 'Bump_Center_Z' in df.columns else ('Z_Coord' if 'Z_Coord' in df.columns else None)
        if z_col and df[z_col].notna().any():
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        elif 'Layer_Number' in df.columns:
            df['Inferred_Layer'] = df['Layer_Number']
        else:
            df['Inferred_Layer'] = 0
        return df

    def calculate_xy_pitch(df):
        """요청사항 1: Height 데이터가 있는 경우에만 Pitch 계산"""
        if 'Height' not in df.columns: return df
        
        x_c = 'Bump_Center_X' if 'Bump_Center_X' in df.columns else ('X_Coord' if 'X_Coord' in df.columns else None)
        y_c = 'Bump_Center_Y' if 'Bump_Center_Y' in df.columns else ('Y_Coord' if 'Y_Coord' in df.columns else None)
        if not x_c or not y_c: return df
        
        res = []
        for l in df['Inferred_Layer'].unique():
            ldf = df[df['Inferred_Layer'] == l].copy()
            if len(ldf) < 2: 
                res.append(ldf); continue
            # X-Pitch (Y 정렬 후 X 차이)
            ldf['Y_G'] = ldf[y_c].round(0)
            ldf = ldf.sort_values(['Y_G', x_c])
            ldf['Pitch_X'] = ldf.groupby('Y_G')[x_c].diff().abs()
            # Y-Pitch (X 정렬 후 Y 차이)
            ldf['X_G'] = ldf[x_c].round(0)
            ldf = ldf.sort_values(['X_G', y_c])
            ldf['Pitch_Y'] = ldf.groupby('X_G')[y_c].diff().abs()
            # IQR 필터 자동 적용
            ldf['Pitch_X'] = apply_iqr_filter(ldf['Pitch_X'])
            ldf['Pitch_Y'] = apply_iqr_filter(ldf['Pitch_Y'])
            res.append(ldf)
        return pd.concat(res) if res else df

    # --- 4. 데이터 로드 및 마스터 매핑 ---

    raw_data = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Master File**을 선택하거나 개별 분석을 위해 **'Independent Analysis'**를 선택하세요.")
    m_options = ["Independent Analysis (No Master)"] + list(raw_data.keys())
    m_key = st.selectbox("Master 파일 선택", m_options)
    
    layer_map, master_coords = None, None
    if m_key != "Independent Analysis (No Master)":
        m_df_proc = get_layer_info(raw_data[m_key], z_gap_threshold)
        layer_map = m_df_proc[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
        # 마스터 좌표 컬럼 자동 선택
        xc_m = 'Bump_Center_X' if 'Bump_Center_X' in m_df_proc.columns else 'X_Coord'
        yc_m = 'Bump_Center_Y' if 'Bump_Center_Y' in m_df_proc.columns else 'Y_Coord'
        zc_m = 'Bump_Center_Z' if 'Bump_Center_Z' in m_df_proc.columns else 'Z_Coord'
        master_coords = m_df_proc[['Group_ID', xc_m, yc_m, zc_m]].rename(columns={xc_m:'X', yc_m:'Y', zc_m:'Z'}).drop_duplicates()

    processed_list = []
    for name, df in raw_data.items():
        if m_key != "Independent Analysis (No Master)" and name != m_key:
            if 'Group_ID' in df.columns:
                df = df.merge(layer_map, on='Group_ID', how='inner')
            else: continue
        else:
            df = get_layer_info(df, z_gap_threshold)
        
        # Pitch 계산 (Height 파일만 자동 필터링)
        df = calculate_xy_pitch(df)
        
        # IQR 필터링
        if use_filter_height and 'Height' in df.columns: df['Height'] = apply_iqr_filter(df['Height'])
        if use_filter_radius and 'Radius' in df.columns: df['Radius'] = apply_iqr_filter(df['Radius'])
        if use_filter_shift:
            for sc in ['Shift_X', 'Shift_Y', 'Shift_Norm']:
                if sc in df.columns: df[sc] = apply_iqr_filter(df[sc])
        
        df['File_Name'] = name
        processed_list.append(df)

    if processed_list:
        full_df = pd.concat(processed_list, ignore_index=True)
        
        # --- 상단 통계 ---
        st.subheader("📊 Summary Statistics")
        m_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
        summary = full_df.groupby(['File_Name', 'Inferred_Layer'])[m_list].agg(['mean', 'std', 'count']).round(3)
        st.dataframe(summary, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📏 Group A: 형상 분석", "🎯 Group B: 위치 편차", "🌐 3D View & Highlight"])
        c_grp = "Inferred_Layer" if "Split" in layer_view_mode else None

        with tab1:
            st.header("Group A: Shape Analysis")
            sel_met_a = st.selectbox("지표 선택 (A)", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in full_df.columns])
            p_df_a = full_df.dropna(subset=[sel_met_a])
            c_a1, c_a2 = st.columns(2)
            with c_a1: st.plotly_chart(px.box(p_df_a, x="File_Name", y=sel_met_a, color=c_grp, points=False, title=f"{sel_met_a} Boxplot"), use_container_width=True)
            with c_a2: st.plotly_chart(px.histogram(p_df_a, x=sel_met_a, color="File_Name" if c_grp is None else c_grp, barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, opacity=0.7), use_container_width=True)

        with tab2:
            st.header("Group B: Alignment Analysis")
            sel_met_b = st.selectbox("Shift 지표 선택 (B)", [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns])
            p_df_b = full_df.dropna(subset=[sel_met_b])
            c_b1, c_b2 = st.columns(2)
            with c_b1: st.plotly_chart(px.box(p_df_b, x="File_Name", y=sel_met_b, color=c_grp, points=False), use_container_width=True)
            with c_b2: st.plotly_chart(px.histogram(p_df_b, x=sel_met_b, color="File_Name" if c_grp is None else c_grp, barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, opacity=0.7), use_container_width=True)
            
            st.divider()
            st.subheader("📍 Shift Vector Map")
            v_file = st.selectbox("Vector Map 파일 선택", p_df_b['File_Name'].unique())
            v_df = p_df_b[p_df_b['File_Name'] == v_file].dropna(subset=['Shift_X', 'Shift_Y'])
            if not v_df.empty:
                xc = 'Bump_Center_X' if 'Bump_Center_X' in v_df.columns else 'X_Coord'
                yc = 'Bump_Center_Y' if 'Bump_Center_Y' in v_df.columns else 'Y_Coord'
                fig_v = ff.create_quiver(x=v_df[xc], y=v_df[yc], u=v_df['Shift_X']*vector_scale, v=v_df['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                fig_v.add_trace(go.Scatter(x=v_df[xc], y=v_df[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3)))
                fig_v.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig_v, use_container_width=True)

        with tab3:
            st.header("🌐 3D Structural Highlight View")
            # 요청사항 2: 마스터 유무에 따른 3D 데이터 로직 수정
            if m_key != "Independent Analysis (No Master)":
                # Master 모드: 모든 지표 통합
                c_3d = st.selectbox("Color/Highlight Metric", [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in full_df.columns])
                pivot_df = full_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
                df3 = pivot_df.merge(master_coords, on='Group_ID', how='left').dropna(subset=[c_3d])
                x3, y3, z3 = 'X', 'Y', 'Z'
            else:
                # Independent 모드: 선택한 파일만 (요청사항 2 해결)
                t_file = st.selectbox("3D 파일 선택", full_df['File_Name'].unique())
                df3 = full_df[full_df['File_Name'] == t_file].copy()
                c_3d = st.selectbox("Color/Highlight Metric", [c for c in df3.columns if df3[c].dtype in ['float64', 'int64']])
                df3 = df3.dropna(subset=[c_3d])
                x3 = 'Bump_Center_X' if 'Bump_Center_X' in df3.columns else ('X_Coord' if 'X_Coord' in df3.columns else 'X')
                y3 = 'Bump_Center_Y' if 'Bump_Center_Y' in df3.columns else ('Y_Coord' if 'Y_Coord' in df3.columns else 'Y')
                z3 = 'Bump_Center_Z' if 'Bump_Center_Z' in df3.columns else ('Z_Coord' if 'Z_Coord' in df3.columns else 'Inferred_Layer')

            # 요청사항 3: 임계값 하이라이트 기능
            st.subheader("⚠️ Threshold Highlighting")
            th_col1, th_col2 = st.columns(2)
            with th_col1:
                high_th = st.number_input(f"High Threshold (Red Above)", value=float(df3[c_3d].max()))
            with th_col2:
                low_th = st.number_input(f"Low Threshold (Yellow Below)", value=float(df3[c_3d].min()))

            # 색상 매핑 로직
            def assign_status_color(val):
                if val >= high_th: return 'Critical (Red)'
                elif val <= low_th: return 'Warning (Yellow)'
                else: return 'Normal'

            df3['Status'] = df3[c_3d].apply(assign_status_color)
            
            fig3 = px.scatter_3d(
                df3, x=x3, y=y3, z=z_3 if 'z_3' in locals() else z3,
                color='Status',
                color_discrete_map={'Critical (Red)': 'red', 'Warning (Yellow)': 'yellow', 'Normal': 'lightgray'},
                opacity=0.8,
                title=f"3D Highlight Map: {c_3d}",
                hover_data=['Group_ID', 'Inferred_Layer', c_3d]
            )
            fig3.update_layout(scene=dict(aspectmode='data'), height=800)
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("데이터 매칭에 실패했습니다.")
else:
    st.info("👈 CSV 파일들을 업로드하세요.")