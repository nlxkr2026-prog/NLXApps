import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Quality Analyzer Pro", layout="wide")
st.title("🔬 Advanced Bump Quality & Multi-Layer Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader(
    "Bump CSV 파일 업로드", 
    type=['csv'], 
    accept_multiple_files=True
)

scale_factor = st.sidebar.selectbox(
    "데이터 단위 변환 (Scale Factor)",
    options=[1, 1000], index=1,
    format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)"
)

z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)

# 레이어 보기 모드 추가 (요청사항 1)
layer_view_mode = st.sidebar.radio(
    "레이어 표시 모드",
    ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"],
    index=0
)

# Pitch Outlier 임계값 설정 (요청사항 2)
st.sidebar.subheader("Pitch Outlier 필터링")
pitch_tolerance = st.sidebar.slider(
    "Pitch 허용 오차 (%)", 
    0, 100, 20, 
    help="평균값에서 이 퍼센트 이상 벗어나면 Outlier로 처리합니다."
)

# --- 3. 핵심 데이터 처리 로직 ---

def preprocess_engine(df, scale, gap):
    """단위 변환 및 층 분리"""
    target_cols = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm']
    for col in df.columns:
        if col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') * scale
            
    if 'Bump_Center_Z' in df.columns:
        df = df.sort_values('Bump_Center_Z').reset_index(drop=True)
        z_diff = df['Bump_Center_Z'].diff().abs()
        df['Inferred_Layer'] = (z_diff > gap).cumsum()
    else:
        df['Inferred_Layer'] = 0
    return df

def calculate_xy_pitch_with_filter(df, tolerance_pct):
    """X, Y Pitch 계산 및 사용자 정의 임계값 기반 Outlier 처리 (요청사항 2)"""
    results = []
    for layer in df['Inferred_Layer'].unique():
        ldf = df[df['Inferred_Layer'] == layer].copy()
        if len(ldf) < 2: continue
        
        # X-Pitch
        ldf['Y_Grid'] = ldf['Bump_Center_Y'].round(0)
        ldf = ldf.sort_values(['Y_Grid', 'Bump_Center_X'])
        ldf['Pitch_X'] = ldf.groupby('Y_Grid')['Bump_Center_X'].diff().abs()
        
        # Y-Pitch
        ldf['X_Grid'] = ldf['Bump_Center_X'].round(0)
        ldf = ldf.sort_values(['X_Grid', 'Bump_Center_Y'])
        ldf['Pitch_Y'] = ldf.groupby('X_Grid')['Bump_Center_Y'].diff().abs()
        
        # Outlier 필터링 로직: 평균 기준 허용 범위 적용
        for p_col in ['Pitch_X', 'Pitch_Y']:
            if p_col in ldf.columns:
                avg_val = ldf[p_col].mean()
                if not np.isnan(avg_val):
                    lower = avg_val * (1 - tolerance_pct / 100)
                    upper = avg_val * (1 + tolerance_pct / 100)
                    # 범위를 벗어나는 데이터는 Outlier(NaN) 처리
                    ldf.loc[(ldf[p_col] < lower) | (ldf[p_col] > upper), p_col] = np.nan
                    
        results.append(ldf)
    return pd.concat(results) if results else df

# --- 4. 메인 실행 및 화면 구성 ---

if uploaded_files:
    all_dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        df = preprocess_engine(df, scale_factor, z_gap_threshold)
        df = calculate_xy_pitch_with_filter(df, pitch_tolerance)
        df['File_Name'] = f.name
        all_dfs.append(df)
    
    master_df = pd.concat(all_dfs, ignore_index=True)

    # --- 요청사항 3: 상단 통계 데이터 대시보드 ---
    st.subheader("📊 데이터 요약 통계 (Summary Statistics)")
    
    # 주요 지표 추출
    stat_metrics = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_Norm'] if c in master_df.columns]
    
    # 파일별 통계 계산
    summary_stats = master_df.groupby('File_Name')[stat_metrics].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    st.dataframe(summary_stats, use_container_width=True)
    
    st.divider()

    # --- 탭 구성 ---
    tab1, tab2, tab3 = st.tabs(["📏 Group A: 형상 & 간격", "🎯 Group B: 위치 편차", "🌐 3D 구조 뷰"])

    # --- Tab 1: Group A 분석 ---
    with tab1:
        st.header("Group A: Shape & Grid Pitch Analysis")
        selected_metric = st.selectbox("분석 지표", stat_metrics)
        
        # 레이어 보기 모드에 따른 시각화 설정 (요청사항 1)
        color_group = "Inferred_Layer" if "Split" in layer_view_mode else None
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{selected_metric} Boxplot**")
            fig_box = px.box(
                master_df, x="File_Name", y=selected_metric, 
                color=color_group, # 층별 분리 여부 결정
                points=False, title=f"Comparison: {selected_metric}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col2:
            st.write(f"**{selected_metric} Histogram**")
            fig_hist = px.histogram(
                master_df, x=selected_metric, 
                color="File_Name" if color_group is None else color_group,
                barmode="overlay", marginal="box", title=f"Distribution: {selected_metric}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- Tab 2: Group B 분석 ---
    with tab2:
        st.header("Group B: Alignment & Shift Analysis")
        if 'Shift_Norm' in master_df.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Shift Norm Boxplot**")
                fig_s_box = px.box(master_df, x="File_Name", y="Shift_Norm", 
                                   color=color_group, points=False)
                st.plotly_chart(fig_s_box, use_container_width=True)
            with c2:
                st.write("**Shift Direction (X-Y Scatter)**")
                fig_s_scatter = px.scatter(master_df, x="Shift_X", y="Shift_Y", 
                                           color="File_Name", opacity=0.4, title="Align Bias Map")
                fig_s_scatter.add_vline(x=0, line_dash="dash")
                fig_s_scatter.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_s_scatter, use_container_width=True)
        else:
            st.warning("Shift 데이터가 없습니다.")

    # --- Tab 3: 3D View ---
    with tab3:
        st.header("3D Layer Visualization")
        target_3d = st.selectbox("3D 뷰 파일 선택", master_df['File_Name'].unique())
        color_3d = st.selectbox("3D 컬러 기준", ["Inferred_Layer", "Radius", "Height", "Pitch_X", "Pitch_Y"])
        
        df_3d = master_df[master_df['File_Name'] == target_3d]
        fig_3d = px.scatter_3d(df_3d, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z',
                               color=color_3d, opacity=0.7, title=f"3D Map: {target_3d}")
        fig_3d.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")