import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer", layout="wide")
st.title("🔬 Bump Quality Multi-Layer Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader("분석할 CSV 파일들을 모두 업로드하세요", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
    z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)
    
    # 레이어 및 히스토그램 설정
    layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])
    hist_layout = st.sidebar.selectbox("히스토그램 레이아웃", ["Facet (파일별 분리)", "Overlay (겹쳐보기)", "Group (나열하기)"])
    
    st.sidebar.subheader("Pitch & Vector 설정")
    pitch_tolerance = st.sidebar.slider("Pitch 허용 오차 (%)", 0, 100, 20)
    vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 200, 50)

    # --- 3. 데이터 처리 함수 ---

    def preprocess_df(df, scale):
        """기본 단위 변환"""
        # 수치로 변환해야 할 대상 컬럼들
        cols = ['Group_ID', 'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
        for c in df.columns:
            if c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if c not in ['Group_ID', 'Inferred_Layer']:
                    df[c] *= scale
        return df

    def get_layer_info(df, gap):
        """Z값을 기준으로 레이어 번호 생성"""
        z_col = 'Bump_Center_Z' if 'Bump_Center_Z' in df.columns else ('Z_Coord' if 'Z_Coord' in df.columns else None)
        if z_col and df[z_col].notna().any():
            # Z값 기준으로 정렬하여 층을 구분
            df = df.sort_values(z_col).reset_index(drop=True)
            df['Inferred_Layer'] = (df[z_col].diff().abs() > gap).cumsum()
        elif 'Layer_Number' in df.columns:
            df['Inferred_Layer'] = df['Layer_Number']
        else:
            df['Inferred_Layer'] = 0
        return df

    def calc_pitch(df, tol):
        """X, Y Pitch 계산"""
        x_c = 'Bump_Center_X' if 'Bump_Center_X' in df.columns else 'X_Coord'
        y_c = 'Bump_Center_Y' if 'Bump_Center_Y' in df.columns else 'Y_Coord'
        if x_c not in df.columns or y_c not in df.columns: return df
        
        res = []
        for l in df['Inferred_Layer'].unique():
            ldf = df[df['Inferred_Layer'] == l].copy()
            if len(ldf) < 2: 
                res.append(ldf)
                continue
            # X-Pitch
            ldf['Y_G'] = ldf[y_c].round(0)
            ldf = ldf.sort_values(['Y_G', x_c])
            ldf['Pitch_X'] = ldf.groupby('Y_G')[x_c].diff().abs()
            # Y-Pitch
            ldf['X_G'] = ldf[x_c].round(0)
            ldf = ldf.sort_values(['X_G', y_c])
            ldf['Pitch_Y'] = ldf.groupby('X_G')[y_c].diff().abs()
            # Outlier Filter
            for p in ['Pitch_X', 'Pitch_Y']:
                avg = ldf[p].mean()
                if not np.isnan(avg):
                    ldf.loc[(ldf[p] < avg*(1-tol/100)) | (ldf[p] > avg*(1+tol/100)), p] = np.nan
            res.append(ldf)
        return pd.concat(res) if res else df

    # --- 4. 메인 로직 실행 ---

    raw_data = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Master File**을 선택하세요. 이 파일의 Group_ID와 Layer 정보를 기준으로 다른 데이터들이 매칭됩니다.")
    master_key = st.selectbox("Master 파일 선택", list(raw_dict.keys()) if 'raw_dict' in locals() else list(raw_data.keys()))
    
    m_df = get_layer_info(raw_data[master_key], z_gap_threshold)
    layer_map = m_df[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
    
    processed_list = []
    for name, df in raw_data.items():
        if name == master_key:
            final_df = m_df
        else:
            if 'Group_ID' in df.columns:
                # Master에 있는 Group_ID만 매칭 (요청대로 마스터에 없는 레이어 데이터는 제외)
                final_df = df.merge(layer_map, on='Group_ID', how='inner')
            else:
                st.warning(f"'{name}' 파일에 Group_ID가 없어 매칭에서 제외되었습니다.")
                continue
        
        final_df = calc_pitch(final_df, pitch_tolerance)
        final_df['File_Name'] = name
        processed_list.append(final_df)

    if processed_list:
        full_df = pd.concat(processed_list, ignore_index=True)

        # --- 상단 통계 (NaN 제외) ---
        st.subheader("📊 Summary Statistics (By File & Layer)")
        metrics = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
        
        # 지표별로 유효한 데이터만 계산되도록 groupby
        summary = full_df.groupby(['File_Name', 'Inferred_Layer'])[metrics].agg(['mean', 'std', 'count']).round(3)
        st.dataframe(summary, use_container_width=True)
        st.divider()

        t1, t2, t3 = st.tabs(["📏 Group A: 형상 & 간격", "🎯 Group B: Align & Shift", "🌐 3D View"])

        # 레이어 보기 모드 설정
        c_grp = "Inferred_Layer" if "Split" in layer_view_mode else None

        with t1:
            st.header("Group A: Shape & Pitch Analysis")
            # 현재 데이터셋에 존재하는 지표만 선택지로 제공
            avail_a = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in full_df.columns]
            sel_met_a = st.selectbox("지표 선택", avail_a)
            
            # 핵심 해결책: 선택한 지표가 NaN인 행은 그래프에서 완전히 제거
            plot_df_a = full_df.dropna(subset=[sel_met_a])
            
            if not plot_df_a.empty:
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    fig_box_a = px.box(plot_df_a, x="File_Name", y=sel_met_a, color=c_grp, points=False, title=f"{sel_met_a} Boxplot")
                    st.plotly_chart(fig_box_a, use_container_width=True)
                with col_a2:
                    b_mode = "overlay" if hist_layout == "Overlay (겹쳐보기)" else "group"
                    f_col = "File_Name" if hist_layout == "Facet (파일별 분리)" else None
                    fig_hist_a = px.histogram(plot_df_a, x=sel_met_a, color="File_Name" if c_grp is None else c_grp, 
                                           barmode=b_mode, facet_col=f_col, opacity=0.7, title=f"{sel_met_a} Distribution")
                    st.plotly_chart(fig_hist_a, use_container_width=True)
            else:
                st.warning(f"'{sel_met_a}' 지표에 유효한 데이터가 없습니다.")

        with t2:
            st.header("Group B: Alignment & Shift Analysis")
            avail_b = [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
            sel_met_b = st.selectbox("Shift 지표 선택", avail_b)
            
            # NaN 제거 후 그래프 생성
            plot_df_b = full_df.dropna(subset=[sel_met_b])
            
            if not plot_df_b.empty:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.plotly_chart(px.box(plot_df_b, x="File_Name", y=sel_met_b, color=c_grp, points=False, title=f"{sel_met_b} Boxplot"), use_container_width=True)
                with col_b2:
                    b_mode = "overlay" if hist_layout == "Overlay (겹쳐보기)" else "group"
                    f_col = "File_Name" if hist_layout == "Facet (파일별 분리)" else None
                    st.plotly_chart(px.histogram(plot_df_b, x=sel_met_b, color="File_Name" if c_grp is None else c_grp, 
                                                barmode=b_mode, facet_col=f_col, opacity=0.7, title=f"{sel_met_b} Distribution"), use_container_width=True)
            
                st.divider()
                st.subheader("📍 Shift Vector Map")
                v_file = st.selectbox("화살표 지도를 볼 파일 선택", plot_df_b['File_Name'].unique())
                v_df = plot_df_b[plot_df_b['File_Name'] == v_file].dropna(subset=['Shift_X', 'Shift_Y'])
                if not v_df.empty:
                    xc = 'Bump_Center_X' if 'Bump_Center_X' in v_df.columns else 'X_Coord'
                    yc = 'Bump_Center_Y' if 'Bump_Center_Y' in v_df.columns else 'Y_Coord'
                    fig_v = ff.create_quiver(x=v_df[xc], y=v_df[yc], u=v_df['Shift_X']*vector_scale, v=v_df['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                    fig_v.add_trace(go.Scatter(x=v_df[xc], y=v_df[yc], mode='markers', marker=dict(size=3, color='blue', opacity=0.3), name='Bump Center'))
                    fig_v.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1), title=f"Vector Map: {v_file} (Scale x{vector_scale})")
                    st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.warning("Shift 유효 데이터가 없습니다.")

        with t3:
            st.header("3D Layer Structural View")
            t_3d = st.selectbox("3D 파일 선택", full_df['File_Name'].unique())
            c_3d = st.selectbox("색상 기준", ["Inferred_Layer", "Radius", "Height", "Pitch_X", "Pitch_Y", "Shift_Norm"])
            df3 = full_df[full_df['File_Name'] == t_3d].copy()
            
            # 색상 지표가 있는 데이터만 3D로 표시
            df3 = df3.dropna(subset=[c_3d]) if c_3d in df3.columns else df3
            
            if not df3.empty:
                zc = 'Bump_Center_Z' if 'Bump_Center_Z' in df3.columns else ('Z_Coord' if 'Z_Coord' in df3.columns else 'Inferred_Layer')
                fig3 = px.scatter_3d(df3, x='Bump_Center_X' if 'Bump_Center_X' in df3.columns else 'X_Coord', 
                                     y='Bump_Center_Y' if 'Bump_Center_Y' in df3.columns else 'Y_Coord', 
                                     z=zc, color=c_3d, opacity=0.7, title=f"3D: {t_3d}")
                fig3.update_layout(scene=dict(aspectmode='data'))
                st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("❌ 매칭된 데이터가 없습니다. Group_ID가 Master 파일과 일치하는지 확인하세요.")
else:
    st.info("👈 왼쪽에서 분석할 Bump CSV 파일들을 모두 업로드해 주세요.")