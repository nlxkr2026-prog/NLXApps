import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer Pro", layout="wide")
st.title("🔬 Universal Bump Quality Analyzer (v.Final_Fixed)")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 및 시각화 설정")

uploaded_files = st.sidebar.file_uploader("모든 CSV 파일을 업로드하세요", type=['csv'], accept_multiple_files=True)

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
        cols_to_scale = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
        for c in df.columns:
            if c in cols_to_scale:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] *= scale
            if c == 'Group_ID':
                df[c] = pd.to_numeric(df[c], errors='coerce')
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

    # --- 4. 메인 데이터 처리 엔진 ---
    raw_data_dict = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    st.info("🎯 **Master File**을 선택하세요. 모든 데이터가 이 기준에 따라 통합됩니다.")
    m_options = ["Independent Analysis"] + list(raw_data_dict.keys())
    m_key = st.selectbox("레이어 기준(Master) 파일 선택", m_options)
    
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

        # --- 5. 데이터 Export 기능 ---
        st.subheader("📁 Data Export")
        m_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in total_df.columns]
        
        summary_by_layer = total_df.groupby(['File_Name', 'Inferred_Layer'])[m_list].agg(['mean', 'std', 'count']).round(3)
        summary_total = total_df.groupby(['File_Name'])[m_list].agg(['mean', 'std', 'count']).round(3)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_by_layer.to_excel(writer, sheet_name='Layer_Statistics')
            summary_total.to_excel(writer, sheet_name='Total_Statistics')
            total_df.to_excel(writer, sheet_name='Raw_Data_Cleaned', index=False)
        
        st.download_button(label="📥 Download Excel Report", data=output.getvalue(), file_name="Bump_Quality_Report.xlsx", mime="application/vnd.ms-excel")

        st.divider()
        st.subheader("📊 Summary Preview")
        st.dataframe(summary_by_layer, use_container_width=True)

        # --- 탭 구성 ---
        tab1, tab2, tab3 = st.tabs(["📏 Group A (Shape)", "🎯 Group B (Shift)", "🌐 3D View"])

        with tab1:
            st.header("Group A: Shape Analysis")
            sel_a = st.selectbox("지표 선택 (A)", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in total_df.columns])
            pdf_a = total_df.dropna(subset=[sel_a])
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.box(pdf_a, x="File_Name", y=sel_a, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False, title=f"{sel_a} Boxplot"), use_container_width=True, config=plot_config)
            c2.plotly_chart(px.histogram(pdf_a, x=sel_a, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None, title=f"{sel_a} Distribution"), use_container_width=True, config=plot_config)
            
            st.divider()
            st.subheader(f"📍 Spatial Heatmap: {sel_a}")
            f_m_a = st.selectbox("지도 출력 파일 선택", pdf_a['File_Name'].unique(), key="ma_f")
            m_a_df = pdf_a[pdf_a['File_Name'] == f_m_a]
            if not m_a_df.empty:
                xc_h, yc_h = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in m_a_df.columns else ('X_Coord', 'Y_Coord')
                fig_heat = px.scatter(m_a_df, x=xc_h, y=yc_h, color=sel_a, facet_col="Inferred_Layer", color_continuous_scale="Turbo")
                fig_heat.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig_heat, use_container_width=True, config=plot_config)

        with tab2:
            st.header("Group B: Shift Analysis")
            b_mets = [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in total_df.columns]
            if b_mets:
                sel_b = st.selectbox("Shift 지표 선택", b_mets)
                pdf_b = total_df.dropna(subset=[sel_b])
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.box(pdf_b, x="File_Name", y=sel_b, color="Inferred_Layer" if "Split" in layer_view_mode else None, points=False), use_container_width=True, config=plot_config)
                c2.plotly_chart(px.histogram(pdf_b, x=sel_b, color="File_Name" if "Split" not in layer_view_mode else "Inferred_Layer", barmode="overlay", facet_col="File_Name" if "Facet" in hist_layout else None), use_container_width=True, config=plot_config)
                
                st.divider()
                st.subheader("📍 Shift Vector Map")
                v_f = st.selectbox("화살표 맵 파일 선택", pdf_b['File_Name'].unique(), key="v_f_b")
                v_d = pdf_b[(pdf_b['File_Name'] == v_f) & pdf_b['Shift_X'].notna()]
                if not v_d.empty:
                    xc, yc = ('Bump_Center_X', 'Bump_Center_Y') if 'Bump_Center_X' in v_d.columns else ('X_Coord', 'Y_Coord')
                    fig_v = ff.create_quiver(x=v_d[xc], y=v_d[yc], u=v_d['Shift_X']*vector_scale, v=v_d['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                    fig_v.add_trace(go.Scatter(x=v_d[xc], y=v_d[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3), name='Bump Center'))
                    fig_v.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
                    st.plotly_chart(fig_v, use_container_width=True, config=plot_config)

        with tab3:
            st.header("🌐 3D Structural View")
            # 3D 통합 데이터 생성
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
                
                # [수정] Threshold 로직 변수 이름 통일 (l_th -> lth)
                apply_th = st.checkbox("⚠️ Threshold Highlight Mode", value=False)
                if apply_th:
                    cx, cy = st.columns(2)
                    hth = cx.number_input("High Threshold (Red Above)", value=float(df3[c_3].max()))
                    lth = cy.number_input("Low Threshold (Yellow Below)", value=float(df3[c_3].min()))
                    
                    # 상태 할당 함수 (변수 이름 오타 수정 완료)
                    def get_status(v):
                        if v >= hth: return 'Critical (Red)'
                        elif v <= lth: return 'Warning (Yellow)'
                        else: return 'Normal'
                    
                    df3['Color_Status'] = df3[c_3].apply(get_status)
                    
                    fig3 = px.scatter_3d(
                        df3, x=x3, y=y3, z=z3, 
                        color='Color_Status',
                        color_discrete_map={'Critical (Red)': 'red', 'Warning (Yellow)': 'yellow', 'Normal': 'lightgray'},
                        opacity=0.8
                    )
                else:
                    fig3 = px.scatter_3d(df3, x=x3, y=y3, z=z3, color=c_3, color_continuous_scale='Turbo', opacity=0.8)
                
                fig3.update_layout(scene=dict(aspectmode='data'), height=800, title=f"3D Map: {c_3} (Editable)")
                st.plotly_chart(fig3, use_container_width=True, config=plot_config)
else:
    st.info("파일을 업로드하면 분석 대시보드가 활성화됩니다.")