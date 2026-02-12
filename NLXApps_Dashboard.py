import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(page_title="Bump Quality Analyzer", layout="wide")
st.title("🔬 Bump Raw Data Multi-Layer Analyzer")
st.markdown("---")

# --- 2. 사이드바 설정 (전처리 엔진) ---
st.sidebar.header("⚙️ Analysis Settings")

# 데이터 업로드
uploaded_files = st.sidebar.file_uploader("Upload Bump CSV Files", type=['csv'], accept_multiple_files=True)

# 단위 스케일링 설정
scale_factor = st.sidebar.selectbox(
    "Select Input Scale Factor (To um)",
    options=[1, 1000],
    format_func=lambda x: "1 (Already um)" if x == 1 else "1000 (mm to um)"
)

# Z-Gap 레이어링 설정
z_gap_threshold = st.sidebar.slider("Z-Gap Layering Threshold (um)", 10, 200, 50)

# --- 3. 핵심 로직 함수 ---

@st.cache_data
def preprocess_data(df, scale, gap):
    # 컬럼 존재 확인 및 스케일링
    cols_to_scale = ['Bump_Center_X', 'Bump_Center_Y', 'Bump_Center_Z', 'Radius', 'Height', 'Shift_X', 'Shift_Y', 'Shift_Norm']
    for col in df.columns:
        if col in cols_to_scale:
            df[col] = df[col] * scale
            
    # Z-Gap 기반 레이어 할당
    if 'Bump_Center_Z' in df.columns:
        df = df.sort_values('Bump_Center_Z')
        z_diff = df['Bump_Center_Z'].diff().abs()
        df['Inferred_Layer'] = (z_diff > gap).cumsum()
    else:
        df['Inferred_Layer'] = 0
    return df

def calculate_pitch_metrics(df):
    if len(df) < 2: return df, 0, 0
    
    # 층별로 Pitch 계산
    results = []
    for layer in df['Inferred_Layer'].unique():
        layer_df = df[df['Inferred_Layer'] == layer].copy()
        if len(layer_df) < 2: continue
        
        coords = layer_df[['Bump_Center_X', 'Bump_Center_Y']].values
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Local Pitch (가장 가까운 이웃과의 거리)
        layer_df['Local_Pitch'] = distances[:, 1]
        
        # Missing Bump 처리 (Median의 1.5배 초과 시 이상치)
        median_pitch = layer_df['Local_Pitch'].median()
        layer_df['Is_Missing_Path'] = layer_df['Local_Pitch'] > (median_pitch * 1.5)
        results.append(layer_df)
        
    return pd.concat(results) if results else df

# --- 4. 메인 데이터 처리 ---

if uploaded_files:
    all_data_list = []
    for f in uploaded_files:
        temp_df = pd.read_csv(f)
        temp_df = preprocess_data(temp_df, scale_factor, z_gap_threshold)
        temp_df['File_Name'] = f.name
        all_data_list.append(temp_df)
    
    master_df = pd.concat(all_data_list, ignore_index=True)

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📊 Group A: Shape & Pitch", "🎯 Group B: Align & Shift", "🌐 Structural 3D View"])

    # --- Tab 1: Shape & Pitch ---
    with tab1:
        st.subheader("Bump Shape & Grid Analysis")
        
        # 지표 필터링 (Group A 성격의 데이터만)
        if 'Radius' in master_df.columns or 'Height' in master_df.columns:
            m_col1, m_col2 = st.columns(2)
            
            # 파일 간 비교 Box Plot
            with m_col1:
                metric = st.selectbox("Select Shape Metric", ["Radius", "Height", "Local_Pitch"])
                if metric == "Local_Pitch":
                    master_df = calculate_pitch_metrics(master_df)
                
                fig_box = px.box(master_df, x="File_Name", y=metric, color="Inferred_Layer", 
                                 points="all", title=f"{metric} Comparison by File")
                st.plotly_chart(fig_box, use_container_width=True)
                
            # Pitch Heatmap (첫 번째 파일 기준)
            with m_col2:
                target_file = st.selectbox("Select File for Heatmap", master_df['File_Name'].unique())
                sub_df = master_df[master_df['File_Name'] == target_file]
                if metric in sub_df.columns:
                    fig_heat = px.scatter(sub_df, x="Bump_Center_X", y="Bump_Center_Y", color=metric,
                                          facet_col="Inferred_Layer", title=f"{target_file} - {metric} Map")
                    st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Radius or Height column not found in uploaded data.")

    # --- Tab 2: Alignment & Shift ---
    with tab2:
        st.subheader("Positional Shift Analysis")
        if 'Shift_Norm' in master_df.columns:
            s_col1, s_col2 = st.columns([1, 1])
            
            with s_col1:
                # Shift Norm 분포 비교
                fig_hist = px.histogram(master_df, x="Shift_Norm", color="File_Name", barmode="overlay",
                                        marginal="box", title="Shift Norm Distribution Comparison")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with s_col2:
                # Vector Scatter (Shift X vs Y)
                fig_shift = px.scatter(master_df, x="Shift_X", y="Shift_Y", color="File_Name",
                                       hover_data=['Group_ID'], title="Shift X-Y Scatter (Align Bias)")
                fig_shift.add_shape(type="circle", x0=-5, y0=-5, x1=5, y1=5, line_color="Red", opacity=0.3)
                st.plotly_chart(fig_shift, use_container_width=True)
        else:
            st.warning("Shift data not found.")

    # --- Tab 3: Structural 3D View ---
    with tab3:
        st.subheader("Integrated 3D Layer Visualization")
        
        c_file = st.selectbox("Select File to view in 3D", master_df['File_Name'].unique())
        c_df = master_df[master_df['File_Name'] == c_file]
        
        color_by = st.selectbox("Color 3D Points by:", ["Inferred_Layer", "Radius", "Height", "Shift_Norm"])
        
        if color_by in c_df.columns:
            fig_3d = px.scatter_3d(
                c_df, x='Bump_Center_X', y='Bump_Center_Y', z='Bump_Center_Z',
                color=color_by, opacity=0.7, size_max=10,
                title=f"3D Map: {c_file} (Colored by {color_by})",
                labels={'Inferred_Layer': 'Layer ID'}
            )
            fig_3d.update_layout(scene=dict(aspectmode='data'))
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)
        else:
            st.error(f"Column '{color_by}' not available for 3D mapping.")

else:
    st.info("👈 Please upload your Bump CSV files in the sidebar to start analysis.")
    st.image("https://img.icons8.com/clouds/500/000000/microchip.png", width=200)