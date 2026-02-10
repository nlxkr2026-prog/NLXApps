import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- [1] 데이터 전처리 및 스마트 레이어 분석 로직 ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 1. 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 2. 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 3. 이상치 제거 (Value)
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 4. [개선] 거시적 레이어 분석 (Z-Gap Detection)
    # 미세한 차이는 무시하고, 0 -> 175 -> 349 같은 거대한 점프만 찾아냅니다.
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['Bump_Center_Z'].unique())
        z_diffs = np.diff(z_vals)
        
        # 전체 Z 범위의 10% 이상이거나, 최소 50 단위 이상 차이나는 곳을 경계로 설정
        z_range = z_vals.max() - z_vals.min()
        gap_threshold = max(z_range * 0.1, 50.0) 
        
        # 경계 지점(Split Points) 찾기
        split_points = z_vals[1:][z_diffs > gap_threshold]
        
        # 레이어 할당 (1부터 시작)
        layer_assignment = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layer_assignment[df_clean['Bump_Center_Z'] >= p] += 1
        df_clean['Layer'] = layer_assignment
    else:
        df_clean['Layer'] = 1

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Advanced Dashboard", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Macro Layering)")

# 사이드바 설정
st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)

# [추가] 그래프 커스터마이징 섹션
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")
custom_x_label = st.sidebar.text_input("X-axis Label", "X Position (um)")
custom_y_label = st.sidebar.text_input("Y-axis Label", "Y Position (um)")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Value Scale")
v_min = st.sidebar.number_input("Value Min", value=0.0)
v_max = st.sidebar.number_input("Value Max", value=20.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, True)
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        
        # 레이어 선택 (1번부터 표시)
        unique_layers = sorted(combined_df['Layer'].unique())
        selected_layer = st.sidebar.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])

        # 필터링 적용
        if selected_layer != "All Layers":
            layer_num = int(selected_layer.split(" ")[1])
            display_df = combined_df[combined_df['Layer'] == layer_num]
        else:
            display_df = combined_df

        # --- 메인 그래프 영역 ---
        st.subheader(f"📊 {d_type} Visual Analysis ({selected_layer})")
        chart_type = st.radio("Select View", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # 모든 그래프에 타이틀/라벨/스케일 적용
        if chart_type == "Heatmap":
            vm_min = v_min if use_custom_scale else display_df['Value'].min()
            vm_max = v_max if use_custom_scale else display_df['Value'].max()
            sc = ax.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], 
                            cmap='jet', s=15, vmin=vm_min, vmax=vm_max)
            plt.colorbar(sc, label=f"{d_type} Value")
            ax.set_xlabel(custom_x_label)
            ax.set_ylabel(custom_y_label)

        elif chart_type == "Box Plot":
            sns.boxplot(data=display_df, x='Source', y='Value', ax=ax)
            ax.set_ylabel(f"{d_type} Value")
            if use_custom_scale: ax.set_ylim(v_min, v_max)
            plt.xticks(rotation=45)

        elif chart_type == "Distribution":
            sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax)
            ax.set_xlabel(f"{d_type} Value")
            if use_custom_scale: ax.set_xlim(v_min, v_max)

        # 전체 공통 제목 적용
        ax.set_title(custom_title)
        st.pyplot(fig)

        # 요약 통계 테이블
        st.markdown("---")
        st.subheader(f"📋 Summary Statistics ({selected_layer})")
        summary_df = display_df.groupby('Source')['Value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        st.dataframe(summary_df, use_container_width=True)