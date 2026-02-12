import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer", layout="wide")
st.title("🔬 Bump Master-Integrated Quality Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 및 필터 설정")

uploaded_files = st.sidebar.file_uploader("분석할 CSV 파일들을 모두 업로드하세요", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
    z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)
    
    st.sidebar.divider()
    st.sidebar.subheader("🚫 Outlier 필터링 (Global)")
    h_min, h_max = st.sidebar.slider("Height 필터 범위 (um)", 0, 500, (5, 200))
    r_min, r_max = st.sidebar.slider("Radius 필터 범위 (um)", 0, 100, (2, 50))
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 시각화 옵션")
    layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])
    hist_layout = st.sidebar.selectbox("히스토그램 레이아웃", ["Facet (파일별 분리)", "Overlay (겹쳐보기)", "Group (나열하기)"])
    
    st.sidebar.divider()
    st.sidebar.subheader("📏 Pitch & Vector")
    pitch_tolerance = st.sidebar.slider("Pitch 허용 오차 (%)", 0, 100, 20)
    vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 200, 50)

    # --- 3. 데이터 처리 함수 ---

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

    def calc_pitch(df, tol):
        x_c = 'Bump_Center_X' if 'Bump_Center_X' in df.columns else 'X_Coord'
        y_c = 'Bump_Center_Y' if 'Bump_Center_Y' in df.columns else 'Y_Coord'
        if x_c not in df.columns or y_c not in df.columns: return df
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
            for p in ['Pitch_X', 'Pitch_Y']:
                avg = ldf[p].mean()
                if not np.isnan(avg):
                    ldf.loc[(ldf[p] < avg*(1-tol/100)) | (ldf[p] > avg*(1+tol/100)), p] = np.nan
            res.append(ldf)
        return pd.concat(res) if res else df

    # --- 4. 메인 로직 실행 ---

    raw_data = {f.name: preprocess_df(pd.read_csv(f), scale_factor) for f in uploaded_files}
    
    st.info("🎯 **Master File**을 선택하세요. 이 파일의 Group_ID를 기준으로 모든 데이터가 통합됩니다.")
    master_key = st.selectbox("Master 파일 선택", list(raw_data.keys()))
    
    m_df = get_layer_info(raw_data[master_key], z_gap_threshold)
    layer_map = m_df[['Group_ID', 'Inferred_Layer']].drop_duplicates().dropna()
    master_coords = m_df[['Group_ID', 'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z']].drop_duplicates()
    
    processed_list = []
    for name, df in raw_data.items():
        if name == master_key:
            final_df = m_df
        else:
            if 'Group_ID' in df.columns:
                final_df = df.merge(layer_map, on='Group_ID', how='inner') # 마스터에 있는 것만 남김
            else: continue
        
        if 'Height' in final_df.columns:
            final_df.loc[(final_df['Height'] < h_min) | (final_df['Height'] > h_max), 'Height'] = np.nan
        if 'Radius' in final_df.columns:
            final_df.loc[(final_df['Radius'] < r_min) | (final_df['Radius'] > r_max), 'Radius'] = np.nan
            
        final_df = calc_pitch(final_df, pitch_tolerance)
        final_df['File_Name'] = name
        processed_list.append(final_df)

    if processed_list:
        full_df = pd.concat(processed_list, ignore_index=True)
        
        # 3D 뷰 및 지표 연동을 위한 통합 테이블 생성 (Pivoting)
        pivot_metrics = full_df.groupby(['Group_ID', 'Inferred_Layer']).first().reset_index()
        integrated_df = pivot_metrics.drop(columns=['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z'], errors='ignore').merge(master_coords, on='Group_ID', how='left')

        # --- 상단 통계 ---
        st.subheader("📊 Summary Statistics (Master-Matched & Filtered)")
        metrics_list = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns]
        summary = full_df.groupby(['File_Name', 'Inferred_Layer'])[metrics_list].agg(['mean', 'std', 'count']).round(3)
        st.dataframe(summary, use_container_width=True)
        st.divider()

        t1, t2, t3 = st.tabs(["📏 Group A: 형상 & 간격", "🎯 Group B: Align & Shift", "🌐 3D 통합 뷰"])
        color_grp = "Inferred_Layer" if "Split" in layer_view_mode else None

        with t1:
            st.header("Group A: Shape Analysis")
            sel_met_a = st.selectbox("지표 선택 (A)", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in full_df.columns])
            plot_df_a = full_df.dropna(subset=[sel_met_a]) # 선택한 지표가 있는 행만 추출 (가시성 해결)
            
            c_a1, c_a2 = st.columns(2)
            with c_a1: st.plotly_chart(px.box(plot_df_a, x="File_Name", y=sel_met_a, color=color_grp, points=False, title=f"{sel_met_a} Boxplot"), use_container_width=True)
            with c_a2:
                b_mode = "overlay" if hist_layout == "Overlay (겹쳐보기)" else "group"
                f_col = "File_Name" if hist_layout == "Facet (파일별 분리)" else None
                st.plotly_chart(px.histogram(plot_df_a, x=sel_met_a, color="File_Name" if color_grp is None else color_grp, barmode=b_mode, facet_col=f_col, opacity=0.7, title=f"{sel_met_a} Distribution"), use_container_width=True)

        with t2:
            st.header("Group B: Alignment Analysis")
            sel_met_b = st.selectbox("Shift 지표 선택 (B)", [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in full_df.columns])
            plot_df_b = full_df.dropna(subset=[sel_met_b])
            
            c_b1, c_b2 = st.columns(2)
            with c_b1: st.plotly_chart(px.box(plot_df_b, x="File_Name", y=sel_met_b, color=color_grp, points=False, title=f"{sel_met_b} Boxplot"), use_container_width=True)
            with c_b2:
                b_mode = "overlay" if hist_layout == "Overlay (겹쳐보기)" else "group"
                f_col = "File_Name" if hist_layout == "Facet (파일별 분리)" else None
                st.plotly_chart(px.histogram(plot_df_b, x=sel_met_b, color="File_Name" if color_grp is None else color_grp, barmode=b_mode, facet_col=f_col, opacity=0.7, title=f"{sel_met_b} Distribution"), use_container_width=True)
            
            st.divider()
            st.subheader("📍 Shift Vector Map")
            v_file = st.selectbox("화살표 맵 파일 선택", plot_df_b['File_Name'].unique())
            v_df = plot_df_b[plot_df_b['File_Name'] == v_file].dropna(subset=['Shift_X', 'Shift_Y'])
            if not v_df.empty:
                fig_v = ff.create_quiver(x=v_df['Bump_Center_X'], y=v_df['Bump_Center_Y'], u=v_df['Shift_X']*vector_scale, v=v_df['Shift_Y']*vector_scale, scale=1, arrow_scale=0.2, line=dict(color='red', width=1))
                fig_v.add_trace(go.Scatter(x=v_df['Bump_Center_X'], y=v_df['Bump_Center_Y'], mode='markers', marker=dict(size=3, color='blue', opacity=0.3)))
                fig_v.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig_v, use_container_width=True)

        with t3:
            st.header("🌐 Integrated 3D Structural View")
            c_3d = st.selectbox("색상 매핑 지표 (Color Mapping)", [c for c in ['Inferred_Layer', 'Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in integrated_df.columns])
            df3 = integrated_df.dropna(subset=[c_3d])
            if not df3.empty:
                fig3 = px.scatter_3d(df3, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z', color=c_3d, opacity=0.8, title=f"3D Map: {c_3d}")
                fig3.update_layout(scene=dict(aspectmode='data'), height=800)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("매칭된 유효 데이터가 없습니다.")