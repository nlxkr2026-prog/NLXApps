import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Quality Analyzer", layout="wide")
st.title("🔬 Universal Bump Quality Analyzer (IQR & Multi-Layer)")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader("분석할 CSV 파일들을 업로드하세요", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
    z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)
    
    st.sidebar.divider()
    st.sidebar.subheader("🛡️ IQR Outlier Filtering")
    # 필터링할 지표 선택 (체크박스)
    filter_radius = st.sidebar.checkbox("Filter Radius Outliers", value=True)
    filter_height = st.sidebar.checkbox("Filter Height Outliers", value=True)
    filter_shift = st.sidebar.checkbox("Filter Shift Outliers", value=False)
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 시각화 옵션")
    layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])
    hist_layout = st.sidebar.selectbox("히스토그램 레이아웃", ["Facet (파일별 분리)", "Overlay (겹쳐보기)"])
    vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 200, 50)

    # --- 3. 데이터 처리 함수 ---

    def apply_iqr_filter(series):
        """IQR 방식을 이용한 이상치 제거 로직"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.mask((series < lower_bound) | (series > upper_bound))

    def preprocess_df(df, scale):
        cols = ['Group_ID', 'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
        for c in df.columns:
            if c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if c != 'Group_ID': df[c] *= scale
        return df

    def get_layer_info(df, gap):
        z_col = 'Bump_Center_Z' if 'Bump_Center_Z' in df.columns else ('Z_Coord' if 'Z_Coord' in df.columns else None)
        if z_col and df[z_col].notna().any():
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        elif 'Layer_Number' in df.columns:
            df['Inferred_Layer'] = df['Layer_Number']
        else:
            df['Inferred_Layer'] = 0
        return df

    # --- 4. 메인 로직 실행 ---

    raw_data = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Master File**을 선택하거나, 각자 분석하려면 **'Independent Analysis'**를 선택하세요.")
    master_options = ["Independent Analysis (No Master)"] + list(raw_data.keys())
    master_key = st.selectbox("Master 파일 선택", master_options)
    
    # 레이어 매핑 테이블 생성
    layer_map = None
    master_coords = None
    if master_key != "Independent Analysis (No Master)":
        m_df = get_layer_info(raw_data[master_key], z_gap_threshold)
        layer_map = m_df[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
        master_coords = m_df[['Group_ID', 'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z']].drop_duplicates()

    processed_list = []
    for name, df in raw_data.items():
        # 레이어링 적용
        if master_key != "Independent Analysis (No Master)" and name != master_key:
            if 'Group_ID' in df.columns:
                df = df.merge(layer_map, on='Group_ID', how='inner')
            else:
                continue # Group_ID가 없으면 매칭 불가하므로 제외
        else:
            df = get_layer_info(df, z_gap_threshold)
            
        # IQR 필터링 적용 (요청 사항)
        if filter_height and 'Height' in df.columns:
            df['Height'] = apply_iqr_filter(df['Height'])
        if filter_radius and 'Radius' in df.columns:
            df['Radius'] = apply_iqr_filter(df['Radius'])
        if filter_shift and 'Shift_Norm' in df.columns:
            df['Shift_Norm'] = apply_iqr_filter(df['Shift_Norm'])
            
        df['File_Name'] = name
        processed_list.append(df)

    if processed_list:
        full_df = pd.concat(processed_list, ignore_index=True)
        
        # --- 상단 통계 ---
        st.subheader("📊 Summary Statistics")
        metrics = [c for c in ['Radius', 'Height', 'Shift_Norm'] if c in full_df.columns]
        summary = full_df.groupby(['File_Name', 'Inferred_Layer'])[metrics].agg(['mean', 'std', 'count']).round(3)
        st.dataframe(summary, use_container_width=True)

        t1, t2, t3 = st.tabs(["📏 Group A: 형상 분석", "🎯 Group B: 위치 편차", "🌐 3D View"])
        c_grp = "Inferred_Layer" if "Split" in layer_view_mode else None

        with t1:
            st.header("Group A: Shape Analysis")
            sel_met_a = st.selectbox("지표 선택 (A)", [c for c in ['Radius', 'Height'] if c in full_df.columns])
            plot_df_a = full_df.dropna(subset=[sel_met_a])
            
            c_a1, c_a2 = st.columns(2)
            with c_a1: st.plotly_chart(px.box(plot_df_a, x="File_Name", y=sel_met_a, color=c_grp, points=False, title=f"{sel_met_a} Boxplot"), use_container_width=True)
            with c_a2:
                fig_h_a = px.histogram(plot_df_a, x=sel_met_a, color="File_Name" if c_grp is None else c_grp, 
                                       barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, opacity=0.7, title=f"{sel_met_a} Distribution")
                st.plotly_chart(fig_h_a, use_container_width=True)

        with t2:
            st.header("Group B: Alignment Analysis")
            sel_met_b = st.selectbox("Shift 지표 선택 (B)", [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns])
            plot_df_b = full_df.dropna(subset=[sel_met_b])
            
            c_b1, c_b2 = st.columns(2)
            with c_b1: st.plotly_chart(px.box(plot_df_b, x="File_Name", y=sel_met_b, color=c_grp, points=False), use_container_width=True)
            with c_b2: st.plotly_chart(px.histogram(plot_df_b, x=sel_met_b, color="File_Name" if c_grp is None else c_grp, 
                                                barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, opacity=0.7), use_container_width=True)
            
            st.divider()
            st.subheader("📍 Shift Vector Map")
            v_file = st.selectbox("Vector Map 파일 선택", plot_df_b['File_Name'].unique())
            v_df = plot_df_b[plot_df_b['File_Name'] == v_file].dropna(subset=['Shift_X', 'Shift_Y'])
            if not v_df.empty:
                xc = 'Bump_Center_X' if 'Bump_Center_X' in v_df.columns else 'X_Coord'
                yc = 'Bump_Center_Y' if 'Bump_Center_Y' in v_df.columns else 'Y_Coord'
                fig_v = ff.create_quiver(x=v_df[xc], y=v_df[yc], u=v_df['Shift_X']*vector_scale, v=v_df['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                fig_v.add_trace(go.Scatter(x=v_df[xc], y=v_df[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3)))
                fig_v.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig_v, use_container_width=True)

        with t3:
            st.header("🌐 3D Structural View")
            if master_key != "Independent Analysis (No Master)":
                # Master 모드: 통합 3D 뷰
                c_3d = st.selectbox("색상 지표", [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Shift_Norm'] if c in full_df.columns])
                pivot_df = full_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
                integrated_df = pivot_df.drop(columns=['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z'], errors='ignore').merge(master_coords, on='Group_ID', how='left')
                df3 = integrated_df.dropna(subset=[c_3d])
                fig3 = px.scatter_3d(df3, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z', color=c_3d, opacity=0.8)
            else:
                # Independent 모드: 선택한 파일만 3D 뷰
                t_file = st.selectbox("3D 파일 선택", full_df['File_Name'].unique())
                df3 = full_df[full_df['File_Name'] == t_file]
                zc = 'Bump_Center_Z' if 'Bump_Center_Z' in df3.columns else ('Z_Coord' if 'Z_Coord' in df3.columns else 'Inferred_Layer')
                fig3 = px.scatter_3d(df3, x='Bump_Center_X' if 'Bump_Center_X' in df3.columns else 'X_Coord', 
                                     y='Bump_Center_Y' if 'Bump_Center_Y' in df3.columns else 'Y_Coord', 
                                     z=zc, color='Inferred_Layer', opacity=0.8)
            fig3.update_layout(scene=dict(aspectmode='data'), height=800)
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("데이터 처리에 실패했습니다. 파일 형식을 확인하세요.")
else:
    st.info("👈 CSV 파일들을 업로드하세요.")