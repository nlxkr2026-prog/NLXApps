import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer Pro", layout="wide")
st.title("🔬 Universal Bump Quality Analyzer (Order Independent)")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 및 시각화 설정")

uploaded_files = st.sidebar.file_uploader("모든 CSV 파일을 업로드하세요 (순서 상관 없음)", type=['csv'], accept_multiple_files=True)

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
    hist_layout = st.sidebar.selectbox("히스토그램 레이아웃", ["Facet (파일별 분할)", "Overlay (겹쳐보기)"])
    vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 200, 50)

    # --- 3. 로직 함수 ---
    def apply_iqr_filter(series):
        if series.dropna().empty: return series
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return series.mask((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR))

    def preprocess_df(df, scale):
        """업로드된 모든 데이터를 um 단위로 표준화"""
        cols_to_scale = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord', 'Top_Z', 'Bottom_Z', 'Middle_Z']
        for c in df.columns:
            if c in cols_to_scale:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] *= scale
            if c == 'Group_ID':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def get_layer_info(df, gap):
        """Z값을 기준으로 레이어 생성 (Master용)"""
        z_col = next((c for c in ['Bump_Center_Z', 'Z_Coord', 'Intersection_Height'] if c in df.columns), None)
        if z_col and df[z_col].notna().any():
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        else:
            df['Inferred_Layer'] = df['Layer_Number'] if 'Layer_Number' in df.columns else 0
        return df

    def calculate_xy_pitch(df):
        """Height 컬럼이 감지될 때만 Pitch 계산"""
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

    # --- 4. 메인 데이터 처리 엔진 (업로드 순서 무시 로직) ---

    # 1단계: 모든 파일 수집
    raw_data_dict = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Master File**을 선택하세요. 어떤 파일을 먼저 올렸든 이 선택이 기준이 됩니다.")
    m_options = ["Independent Analysis"] + list(raw_data_dict.keys())
    m_key = st.selectbox("레이어 기준(Master) 파일 선택", m_options)
    
    # 2단계: 기준 정보 추출
    layer_map, master_coords = None, None
    if m_key != "Independent Analysis":
        m_df_base = get_layer_info(raw_data_dict[m_key], z_gap_threshold)
        layer_map = m_df_base[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
        xc_m = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in m_df_base.columns), 'X')
        yc_m = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in m_df_base.columns), 'Y')
        zc_m = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in m_df_base.columns), 'Z')
        master_coords = m_df_base[['Group_ID', xc_m, yc_m, zc_m]].rename(columns={xc_m:'X', yc_m:'Y', zc_m:'Z'}).drop_duplicates()

    # 3단계: 통합 처리
    final_processed_list = []
    for name, df in raw_data_dict.items():
        # 레이어 매핑 (Master 기준)
        if m_key != "Independent Analysis" and name != m_key:
            if 'Group_ID' in df.columns:
                df = df.merge(layer_map, on='Group_ID', how='inner')
            else: continue
        else:
            df = get_layer_info(df, z_gap_threshold)
        
        # Pitch 및 필터링
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

        # --- 대시보드 출력 ---
        st.subheader("📊 Summary Statistics")
        m_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in total_df.columns]
        st.dataframe(total_df.groupby(['File_Name', 'Inferred_Layer'])[m_list].agg(['mean', 'std', 'count']).round(3), use_container_width=True)

        t1, t2, t3 = st.tabs(["📏 Group A (Shape)", "🎯 Group B (Shift)", "🌐 3D View"])

        with t1:
            sel_a = st.selectbox("지표 선택", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in total_df.columns])
            pdf_a = total_df.dropna(subset=[sel_a])
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.box(pdf_a, x="File_Name", y=sel_a, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True)
            c2.plotly_chart(px.histogram(pdf_a, x=sel_a, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True)
            
            st.divider()
            st.subheader("📍 Spatial Heatmap")
            f_m = st.selectbox("지도 파일", total_df['File_Name'].unique(), key="ma")
            m_a = total_df[(total_df['File_Name'] == f_m) & (total_df[sel_a].notna())]
            xc_a = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in m_a.columns), 'X')
            yc_a = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in m_a.columns), 'Y')
            st.plotly_chart(px.scatter(m_a, x=xc_a, y=yc_a, color=sel_a, facet_col="Inferred_Layer", color_continuous_scale="Turbo"), use_container_width=True)

        with t2:
            b_mets = [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in total_df.columns]
            if b_mets:
                sel_b = st.selectbox("Shift 선택", b_mets)
                pdf_b = total_df.dropna(subset=[sel_b])
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.box(pdf_b, x="File_Name", y=sel_b, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True)
                c2.plotly_chart(px.histogram(pdf_b, x=sel_b, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True)
                
                if 'Shift_X' in total_df.columns and 'Shift_Y' in total_df.columns:
                    st.divider()
                    st.subheader("📍 Shift Vector Map")
                    v_f = st.selectbox("화살표 파일", total_df['File_Name'].unique())
                    v_d = total_df[(total_df['File_Name'] == v_f) & total_df['Shift_X'].notna()]
                    if not v_d.empty:
                        xc, yc = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in v_d.columns else ('X_Coord', 'Y_Coord')
                        fig_v = ff.create_quiver(x=v_d[xc], y=v_d[yc], u=v_d['Shift_X']*vector_scale, v=v_d['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                        fig_v.add_trace(go.Scatter(x=v_d[xc], y=v_d[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3)))
                        fig_v.update_layout(height=700, yaxis=dict(scaleanchor="x", scaleratio=1))
                        st.plotly_chart(fig_v, use_container_width=True)

        with t3:
            st.header("🌐 3D Structural View")
            if m_key != "Independent Analysis":
                pivot_df = total_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
                df3 = pivot_df.merge(master_coords, on='Group_ID', how='left')
                x3, y3, z3 = 'X', 'Y', 'Z'
            else:
                t_3 = st.selectbox("3D 파일 선택", total_df['File_Name'].unique())
                df3 = total_df[total_df['File_Name'] == t_3].copy()
                x3 = next((c for c in ['Bump_Center_X', 'X_Coord'] if c in df3.columns), 'X')
                y3 = next((c for c in ['Bump_Center_Y', 'Y_Coord'] if c in df3.columns), 'Y')
                z3 = next((c for c in ['Bump_Center_Z', 'Z_Coord'] if c in df3.columns), 'Inferred_Layer')

            avail_3d = [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in df3.columns]
            if avail_3d:
                c_3 = st.selectbox("3D 색상 지표", avail_3d)
                df3 = df3.dropna(subset=[c_3])
                apply_th = st.checkbox("⚠️ Threshold Highlight", value=False)
                if apply_th:
                    cx, cy = st.columns(2)
                    hth = cx.number_input("High (Red)", value=float(df3[c_3].max()))
                    lth = cy.number_input("Low (Yellow)", value=float(df3[c_3].min()))
                    df3['Color'] = df3[c_3].apply(lambda v: 'Red' if v >= hth else ('Yellow' if v <= lth else 'Normal'))
                    fig3 = px.scatter_3d(df3, x=x3, y=y3, z=z3, color='Color', color_discrete_map={'Red': 'red', 'Yellow': 'yellow', 'Normal': 'lightgray'})
                else:
                    fig3 = px.scatter_3d(df3, x=x3, y=y3, z=z3, color=c_3, color_continuous_scale='Turbo')
                fig3.update_layout(scene=dict(aspectmode='data'), height=800)
                st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("CSV 파일들을 업로드하세요. 업로드 순서는 상관없습니다.")