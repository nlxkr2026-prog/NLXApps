import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(page_title="Bump Quality Analyzer", layout="wide")
st.title("🔬 Advanced Bump Raw Data Multi-Analyzer")
st.markdown("""
이 도구는 Bump의 형상, 위치 정밀도 및 적층 구조를 분석합니다. 
여러 파일을 업로드하여 공정 간 편차를 비교하고 3D로 시각화할 수 있습니다.
""")

# --- 2. 사이드바 제어판 (전처리 및 엔진 설정) ---
st.sidebar.header("⚙️ 분석 설정 (Global Settings)")

# 데이터 업로드
uploaded_files = st.sidebar.file_uploader(
    "Bump CSV 파일 업로드 (여러 개 가능)", 
    type=['csv'], 
    accept_multiple_files=True
)

# 단위 변환 배수 설정
scale_factor = st.sidebar.selectbox(
    "데이터 단위 변환 (Scale Factor)",
    options=[1, 1000],
    index=1,
    format_func=lambda x: "1 (이미 um 단위)" if x == 1 else "1000 (mm -> um 변환)"
)

# 레이어 분리 임계값
z_gap_threshold = st.sidebar.slider(
    "층 분리 Z-Gap 임계값 (um)", 
    min_value=10, 
    max_value=500, 
    value=50,
    help="Z축 좌표 차이가 이 값보다 크면 새로운 층으로 구분합니다."
)

# --- 3. 핵심 데이터 처리 로직 ---

def preprocess_engine(df, scale, gap):
    """단위 변환 및 Z-Gap 기반 층 분리 로직"""
    # 수치형 변환 및 스케일링 적용
    target_cols = [
        'Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 
        'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm',
        'Top_Z', 'Bottom_Z'
    ]
    for col in df.columns:
        if col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') * scale
            
    # Z-Gap 기반 레이어 할당
    if 'Bump_Center_Z' in df.columns:
        df = df.sort_values('Bump_Center_Z').reset_index(drop=True)
        z_diff = df['Bump_Center_Z'].diff().abs()
        df['Inferred_Layer'] = (z_diff > gap).cumsum()
    else:
        df['Inferred_Layer'] = 0
        
    return df

def calculate_xy_pitch(df):
    """X, Y 방향별 Pitch 계산 및 Missing Bump(이상치) 필터링"""
    results = []
    for layer in df['Inferred_Layer'].unique():
        ldf = df[df['Inferred_Layer'] == layer].copy()
        if len(ldf) < 2: continue
        
        # X-Pitch 계산 (Y가 유사한 행끼리 그룹화)
        ldf['Y_Grid'] = ldf['Bump_Center_Y'].round(0) # 1um 단위 그리드 정렬
        ldf = ldf.sort_values(['Y_Grid', 'Bump_Center_X'])
        ldf['Pitch_X'] = ldf.groupby('Y_Grid')['Bump_Center_X'].diff().abs()
        
        # Y-Pitch 계산 (X가 유사한 열끼리 그룹화)
        ldf['X_Grid'] = ldf['Bump_Center_X'].round(0)
        ldf = ldf.sort_values(['X_Grid', 'Bump_Center_Y'])
        ldf['Pitch_Y'] = ldf.groupby('X_Grid')['Bump_Center_Y'].diff().abs()
        
        # Missing Bump Guard: Median의 1.5배 초과 시 통계 제외 (NaN 처리)
        for p_col in ['Pitch_X', 'Pitch_Y']:
            if p_col in ldf.columns:
                med = ldf[p_col].median()
                if not np.isnan(med):
                    ldf.loc[ldf[p_col] > med * 1.5, p_col] = np.nan
                    
        results.append(ldf)
    return pd.concat(results) if results else df

# --- 4. 메인 대시보드 실행 ---

if uploaded_files:
    # 모든 파일 처리 및 통합
    all_data = []
    for f in uploaded_files:
        raw_df = pd.read_csv(f)
        proc_df = preprocess_engine(raw_df, scale_factor, z_gap_threshold)
        proc_df = calculate_xy_pitch(proc_df)
        proc_df['File_Name'] = f.name
        all_data.append(proc_df)
    
    master_df = pd.concat(all_data, ignore_index=True)

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📊 Group A: Shape & Pitch", "🎯 Group B: Align & Shift", "🌐 3D Layer View"])

    # --- Tab 1: Group A (형상 및 간격) ---
    with tab1:
        st.header("Bump 형상 및 그리드 간격 분석")
        
        available_metrics = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in master_df.columns]
        selected_metric = st.selectbox("비교 분석 지표 선택", available_metrics)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("파일별 분포 비교 (Boxplot)")
            # 요청사항: 단순한 Boxplot (Scattering 제거)
            fig_box = px.box(
                master_df, x="File_Name", y=selected_metric, color="Inferred_Layer",
                points=False, # 점 제거
                title=f"File Comparison: {selected_metric}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col2:
            st.subheader("파일별 밀도 비교 (Histogram)")
            # 요청사항: 히스토그램 추가
            fig_hist = px.histogram(
                master_df, x=selected_metric, color="File_Name",
                barmode="overlay", marginal="rug",
                title=f"Distribution: {selected_metric}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        st.subheader("Spatial Heatmap (공간 분포)")
        target_f = st.selectbox("상세 지도를 볼 파일 선택", master_df['File_Name'].unique())
        f_df = master_df[master_df['File_Name'] == target_f]
        fig_map = px.scatter(
            f_df, x="Bump_Center_X", y="Bump_Center_Y", color=selected_metric,
            facet_col="Inferred_Layer",
            color_continuous_scale="Viridis",
            title=f"{target_f} - {selected_metric} 위치별 분포"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # --- Tab 2: Group B (위치 정밀도) ---
    with tab2:
        st.header("Position Shift 분석 (정렬 오차)")
        if 'Shift_Norm' in master_df.columns:
            b_col1, b_col2 = st.columns(2)
            
            with b_col1:
                st.subheader("Shift Norm 비교")
                st.plotly_chart(px.box(master_df, x="File_Name", y="Shift_Norm", points=False), use_container_width=True)
                
            with b_col2:
                st.subheader("Shift Bias (X-Y Scatter)")
                fig_scatter = px.scatter(
                    master_df, x="Shift_X", y="Shift_Y", color="File_Name",
                    hover_data=['Group_ID'], opacity=0.5,
                    title="Shift X vs Shift Y (쏠림 방향)"
                )
                # 중심점 가이드라인
                fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("데이터에 Shift 관련 컬럼이 없습니다.")

    # --- Tab 3: Structural 3D View ---
    with tab3:
        st.header("3D 적층 구조 시각화")
        
        view_f = st.selectbox("3D 뷰어 파일 선택", master_df['File_Name'].unique(), key="3d_sel")
        view_df = master_df[master_df['File_Name'] == view_f]
        
        color_target = st.selectbox("3D 컬러 기준", ["Inferred_Layer", "Radius", "Height", "Pitch_X", "Pitch_Y"])
        
        if color_target in view_df.columns:
            fig_3d = px.scatter_3d(
                view_df, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z',
                color=color_target, 
                size_max=8, opacity=0.8,
                title=f"3D View: {view_f} (Colored by {color_target})"
            )
            # 실제 비율 유지를 위한 aspectmode 설정
            fig_3d.update_layout(scene=dict(aspectmode='data'))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.error(f"선택한 '{color_target}' 데이터가 파일에 존재하지 않습니다.")

else:
    # 파일 업로드 전 초기 화면
    st.info("👈 왼쪽 사이드바에서 분석할 Bump CSV 파일들을 업로드해 주세요.")
    st.image("https://img.icons8.com/clouds/500/000000/microchip.png", width=150)
    st.markdown("""
    ### 사용 방법
    1. **CSV 업로드**: 분석 대상인 하나 이상의 파일을 올립니다.
    2. **Scale 설정**: $mm$ 단위 데이터라면 `1000`을 선택하세요.
    3. **Z-Gap 조절**: 3D 뷰 탭에서 층이 잘 나뉘는지 확인하며 슬라이더를 조절하세요.
    4. **탭 이동**: 형상(Radius), 간격(Pitch), 위치오차(Shift)를 확인하세요.
    """)