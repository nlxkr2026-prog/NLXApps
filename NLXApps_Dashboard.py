import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Bump Master Analyzer", layout="wide")
st.title("🔬 Bump Master-Sourcing & Multi-Layer Analyzer")

# --- 2. 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정 (Settings)")

uploaded_files = st.sidebar.file_uploader("모든 Bump CSV 파일 업로드", type=['csv'], accept_multiple_files=True)
scale_factor = st.sidebar.selectbox("단위 변환 (Scale Factor)", [1, 1000], index=1, format_func=lambda x: "1 (um)" if x == 1 else "1000 (mm -> um)")
z_gap_threshold = st.sidebar.slider("Z-Gap 레이어링 임계값 (um)", 10, 500, 50)
layer_view_mode = st.sidebar.radio("레이어 표시 모드", ["전체 통합 (Layer All)", "레이어별 분리 (Split by Layer)"])

st.sidebar.subheader("Pitch & Vector 설정")
pitch_tolerance = st.sidebar.slider("Pitch 허용 오차 (%)", 0, 100, 20)
vector_scale = st.sidebar.slider("화살표 배율 (Vector Scale)", 1, 100, 20)

# --- 3. 핵심 로직 함수 ---

def preprocess_basic(df, scale):
    """기본 단위 변환 및 수치화"""
    target_cols = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm', 'X_Coord', 'Y_Coord', 'Z_Coord']
    for col in df.columns:
        if col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') * scale
    return df

def apply_layering(df, gap):
    """Z값이 있는 경우 레이어 자동 생성"""
    z_col = 'Bump_Center_Z' if 'Bump_Center_Z' in df.columns else 'Z_Coord'
    if z_col in df.columns and df[z_col].notna().any():
        df = df.sort_values(z_col).reset_index(drop=True)
        z_diff = df[z_col].diff().abs()
        df['Inferred_Layer'] = (z_diff > gap).cumsum()
    elif 'Layer_Number' in df.columns:
        df['Inferred_Layer'] = df['Layer_Number']
    else:
        df['Inferred_Layer'] = 0
    return df

def calculate_xy_pitch(df, tolerance_pct):
    """X, Y Pitch 계산 및 Outlier 필터링"""
    results = []
    # Pitch 계산에 필요한 컬럼 매핑 (X_Coord 등 대응)
    x_col = 'Bump_Center_X' if 'Bump_Center_X' in df.columns else 'X_Coord'
    y_col = 'Bump_Center_Y' if 'Bump_Center_Y' in df.columns else 'Y_Coord'
    
    if x_col not in df.columns or y_col not in df.columns:
        return df

    for layer in df['Inferred_Layer'].unique():
        ldf = df[df['Inferred_Layer'] == layer].copy()
        if len(ldf) < 2: continue
        
        ldf['Y_Grid'] = ldf[y_col].round(0)
        ldf = ldf.sort_values(['Y_Grid', x_col])
        ldf['Pitch_X'] = ldf.groupby('Y_Grid')[x_col].diff().abs()
        
        ldf['X_Grid'] = ldf[x_col].round(0)
        ldf = ldf.sort_values(['X_Grid', y_col])
        ldf['Pitch_Y'] = ldf.groupby('X_Grid')[y_col].diff().abs()
        
        for p_col in ['Pitch_X', 'Pitch_Y']:
            avg = ldf[p_col].mean()
            if not np.isnan(avg):
                ldf.loc[(ldf[p_col] < avg*(1-tolerance_pct/100)) | (ldf[p_col] > avg*(1+tolerance_pct/100)), p_col] = np.nan
        results.append(ldf)
    return pd.concat(results) if results else df

# --- 4. 메인 데이터 처리 ---

if uploaded_files:
    # 1단계: 모든 파일 로드 및 기본 스케일링
    raw_dict = {}
    for f in uploaded_files:
        raw_dict[f.name] = preprocess_basic(pd.read_csv(f), scale_factor)
    
    # 2단계: 마스터 파일 지정 (레이어 기준점)
    st.info("💡 레이어 기준이 되는 파일(Master)을 선택해 주세요. (예: Height 또는 Multi-layer 파일)")
    master_file_name = st.selectbox("Master 파일 선택", list(raw_dict.keys()))
    
    # 3단계: 마스터 파일 레이어링 수행 및 매핑 테이블 생성
    master_df = apply_layering(raw_dict[master_file_name], z_gap_threshold)
    layer_map = master_df[['Group_ID', 'Inferred_Layer']].drop_duplicates()
    
    # 4단계: 전체 파일에 레이어 매핑 및 Pitch 계산
    final_dfs = []
    for name, df in raw_dict.items():
        if name == master_file_name:
            processed_df = master_df
        else:
            # Group_ID를 기준으로 레이어 정보 병합
            if 'Group_ID' in df.columns:
                df = df.merge(layer_map, on='Group_ID', how='left')
                df['Inferred_Layer'] = df['Inferred_Layer'].fillna(0)
            else:
                df['Inferred_Layer'] = 0
            processed_df = df
        
        processed_df = calculate_xy_pitch(processed_df, pitch_tolerance)
        processed_df['File_Name'] = name
        final_dfs.append(processed_df)
    
    combined_df = pd.concat(final_dfs, ignore_index=True)

    # --- 요청 반영 1 & 3: 상단 통계 ---
    st.subheader("📊 Summary Statistics (Mapped by Master Layer)")
    stat_metrics = [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y', 'Shift_X', 'Shift_Y', 'Shift_Norm'] if c in combined_df.columns]
    summary = combined_df.groupby(['File_Name', 'Inferred_Layer'])[stat_metrics].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(summary, use_container_width=True)
    st.divider()

    # --- 탭 구성 ---
    tab1, tab2, tab3 = st.tabs(["📏 Group A: 형상 & 간격", "🎯 Group B: Align & Shift", "🌐 3D 구조 뷰"])

    with tab1:
        st.header("Group A: Shape & Grid Analysis")
        met_a = st.selectbox("지표 선택", [c for c in ['Radius', 'Height', 'Pitch_X', 'Pitch_Y'] if c in combined_df.columns])
        color_grp = "Inferred_Layer" if "Split" in layer_view_mode else None
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(combined_df, x="File_Name", y=met_a, color=color_grp, points=False), use_container_width=True)
        with c2: st.plotly_chart(px.histogram(combined_df, x=met_a, color="File_Name" if color_grp is None else color_grp, barmode="overlay"), use_container_width=True)

    with tab2:
        st.header("Group B: Shift & Vector Analysis")
        met_b = st.selectbox("Shift 지표", [c for c in ['Shift_X', 'Shift_Y', 'Shift_Norm'] if c in combined_df.columns])
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(combined_df, x="File_Name", y=met_b, color=color_grp, points=False), use_container_width=True)
        with c2: st.plotly_chart(px.histogram(combined_df, x=met_b, color="File_Name" if color_grp is None else color_grp, barmode="overlay"), use_container_width=True)
        
        st.divider()
        st.subheader("📍 Inner Bump Shift Vector Map")
        sel_f = st.selectbox("Vector Map용 파일 선택", combined_df['File_Name'].unique())
        f_df = combined_df[combined_df['File_Name'] == sel_f].dropna(subset=['Shift_X', 'Shift_Y'])
        
        if not f_df.empty:
            x_c = 'Bump_Center_X' if 'Bump_Center_X' in f_df.columns else 'X_Coord'
            y_c = 'Bump_Center_Y' if 'Bump_Center_Y' in f_df.columns else 'Y_Coord'
            fig_v = ff.create_quiver(x=f_df[x_c], y=f_df[y_c], u=f_df['Shift_X']*vector_scale, v=f_df['Shift_Y']*vector_scale, scale=1, name='Shift Vector', line=dict(color='red'))
            fig_v.add_trace(go.Scatter(x=f_df[x_c], y=f_df[y_c], mode='markers', marker=dict(size=3, color='blue', opacity=0.4)))
            fig_v.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(fig_v, use_container_width=True)

    with tab3:
        st.header("3D Structural View")
        target_3d = st.selectbox("3D 파일", combined_df['File_Name'].unique())
        color_3d = st.selectbox("Color Mapping", ["Inferred_Layer", "Radius", "Height", "Pitch_X", "Pitch_Y", "Shift_Norm"])
        d3 = combined_df[combined_df['File_Name'] == target_3d]
        z_c = 'Bump_Center_Z' if 'Bump_Center_Z' in d3.columns else ('Z_Coord' if 'Z_Coord' in d3.columns else 'Inferred_Layer')
        fig3 = px.scatter_3d(d3, x='Bump_Center_X' if 'Bump_Center_X' in d3.columns else 'X_Coord', 
                             y='Bump_Center_Y' if 'Bump_Center_Y' in d3.columns else 'Y_Coord', 
                             z=z_c, color=color_3d, opacity=0.7)
        fig3.update_layout(scene=dict(aspectmode='data'))
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("CSV 파일들을 업로드하고 Master 파일을 지정해 주세요.")