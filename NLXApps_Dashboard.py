import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Quality Analyzer Pro", layout="wide")
st.title("🔬 Advanced Bump Quality & Vector Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader("Bump CSV 파일 업로드", type=['csv'], accept_multiple_files=True)
scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)

# 레이어 보기 모드
layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])

# Pitch & Vector 설정
st.sidebar.subheader("Pitch & Vector 설정")
pitch_tolerance = st.sidebar.slider("Pitch 허용 오차 (%)", 0, 100, 20)
vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 100, 20)

# --- 3. 로직 함수 ---

def preprocess_engine(df, scale, gap, manual_layer=None):
    """단위 변환 및 층 분리 (Z값이 없을 경우 manual_layer 적용)"""
    target_cols = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm']
    for col in df.columns:
        if col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') * scale
            
    if 'Bump_Center_Z' in df.columns and df['Bump_Center_Z'].notna().any():
        df = df.sort_values('Bump_Center_Z').reset_index(drop=True)
        z_diff = df['Bump_Center_Z'].diff().abs()
        df['Inferred_Layer'] = (z_diff > gap).cumsum()
    else:
        # Z값이 없는 경우 사용자가 지정한 층 혹은 0번 할당
        df['Inferred_Layer'] = manual_layer if manual_layer is not None else 0
    return df

def calculate_xy_pitch(df, tolerance_pct):
    results = []
    for layer in df['Inferred_Layer'].unique():
        ldf = df[df['Inferred_Layer'] == layer].copy()
        if len(ldf) < 2: continue
        ldf['Y_Grid'] = ldf['Bump_Center_Y'].round(0)
        ldf = ldf.sort_values(['Y_Grid', 'Bump_Center_X'])
        ldf['Pitch_X'] = ldf.groupby('Y_Grid')['Bump_Center_X'].diff().abs()
        
        ldf['X_Grid'] = ldf['Bump_Center_X'].round(0)
        ldf = ldf.sort_values(['X_Grid', 'Bump_Center_Y'])
        ldf['Pitch_Y'] = ldf.groupby('X_Grid')['Bump_Center_Y'].diff().abs()
        
        for p_col in ['Pitch_X', 'Pitch_Y']:
            if p_col in ldf.columns:
                avg = ldf[p_col].mean()
                if not np.isnan(avg):
                    ldf.loc[(ldf[p_col] < avg*(1-tolerance_pct/100)) | (ldf[p_col] > avg*(1+tolerance_pct/100)), p_col] = np.nan
        results.append(ldf)
    return pd.concat(results) if results else df

# --- 4. 메인 데이터 로드 및 처리 ---

if uploaded_files:
    all_dfs = []
    
    # Z값이 없는 파일을 위해 사이드바에 수동 레이어 지정 UI 생성 (파일별)
    st.sidebar.subheader("파일별 레이어 수동 지정")
    manual_layers = {}
    for f in uploaded_files:
        # 파일명에 힌트가 있는지 확인 (예: layer1, L2 등)
        default_l = 0
        if 'layer' in f.name.lower():
            try: default_l = int(''.join(filter(str.isdigit, f.name))) 
            except: default_l = 0
        manual_layers[f.name] = st.sidebar.number_input(f"{f.name} Layer", 0, 10, default_l)

    for f in uploaded_files:
        df = pd.read_csv(f)
        df = preprocess_engine(df, scale_factor, z_gap_threshold, manual_layer=manual_layers[f.name])
        df = calculate_xy_pitch(df, pitch_tolerance)
        df['File_Name'] = f.name
        all_dfs.append(df)
    
    master_df = pd.concat(all_dfs, ignore_index=True)

    # --- 요청 1 & 3: Layer별 통계 포함 Summary Statistics ---
    st.subheader("📊 Summary Statistics (by File & Layer)")
    stat_metrics = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in master_df.columns]
    summary_stats = master_df.groupby(['File_Name', 'Inferred_Layer'])[stat_metrics].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(summary_stats, use_container_width=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📏 Group A: 형상 & 간격", "🎯 Group B: Align & Shift", "🌐 3D 구조 뷰"])

    with tab1:
        st.header("Group A: Shape & Grid Analysis")
        met_a = st.selectbox("분석 지표 선택", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in master_df.columns])
        color_grp = "Inferred_Layer" if "Split" in layer_view_mode else None
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(master_df, x="File_Name", y=met_a, color=color_grp, points=False, title=f"{met_a} Boxplot"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(master_df, x=met_a, color="File_Name" if color_grp is None else color_grp, barmode="overlay", title=f"{met_a} Histogram"), use_container_width=True)

    with tab2:
        st.header("Group B: Shift & Vector Analysis")
        # 요청 2: Shift X, Y, Norm 선택
        met_b = st.selectbox("Shift 지표 선택", [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in master_df.columns])
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(master_df, x="File_Name", y=met_b, color=color_grp, points=False, title=f"{met_b} Boxplot"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(master_df, x=met_b, color="File_Name" if color_grp is None else color_grp, barmode="overlay", title=f"{met_b} Histogram"), use_container_width=True)
        
        st.divider()
        # 요청 3: Shift Vector Map (화살표 Plot)
        st.subheader("📍 Shift Vector Map (Inner Bump Shift)")
        sel_f = st.selectbox("Vector Map 파일 선택", master_df['File_Name'].unique())
        f_df = master_df[master_df['File_Name'] == sel_f].dropna(subset=['Shift_X', 'Shift_Y'])
        
        if not f_df.empty:
            # Vector Map (Quiver Plot)
            fig_vector = ff.create_quiver(
                x=f_df['Bump_Center_X'], y=f_df['Bump_Center_Y'],
                u=f_df['Shift_X'] * vector_scale, v=f_df['Shift_Y'] * vector_scale,
                scale=1, arrow_scale=0.3, name='Shift Vector', line=dict(width=1, color='red')
            )
            fig_vector.add_trace(go.Scatter(x=f_df['Bump_Center_X'], y=f_df['Bump_Center_Y'], mode='markers', marker=dict(size=3, color='blue', opacity=0.4), name='Bump Center'))
            fig_vector.update_layout(title=f"Shift Direction Map (Scale: x{vector_scale})", height=800)
            fig_vector.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_vector, use_container_width=True)

    with tab3:
        st.header("3D Structural View")
        target_3d = st.selectbox("3D 파일 선택", master_df['File_Name'].unique(), key="3d_sel")
        color_3d = st.selectbox("Color Mapping", ["Inferred_Layer", "Radius", "Height", "Pitch_X", "Pitch_Y", "Shift_Norm"])
        d3 = master_df[master_df['File_Name'] == target_3d]
        fig3 = px.scatter_3d(d3, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z' if 'Bump_Center_Z' in d3.columns else 'Inferred_Layer', color=color_3d, opacity=0.7)
        fig3.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")