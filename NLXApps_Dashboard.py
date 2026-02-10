import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 및 스마트 레이어 분석 ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명을 모두 대문자로 변환하여 대소문자 문제 해결
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 데이터 타입 판별 및 타겟 설정 (X_COORD, HEIGHT, RADIUS 등 대응)
    if 'HEIGHT' in df.columns: d_type, target = "Height", "HEIGHT"
    elif 'RADIUS' in df.columns: d_type, target = "Radius", "RADIUS"
    elif 'SHIFT_NORM' in df.columns: d_type, target = "Shift", "SHIFT_NORM"
    elif 'X_COORD' in df.columns: d_type, target = "Coordinate", "X_COORD"
    else: return None, None

    # 기본 단위 변환 및 좌표 설정 (X_COORD 또는 BUMP_CENTER_X 대응)
    df['X'] = (df['X_COORD'] if 'X_COORD' in df.columns else df['BUMP_CENTER_X']) * scale_factor
    df['Y'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df['BUMP_CENTER_Y']) * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # IQR 필터링
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 레이어 분석 (Z-Gap 또는 LAYER 컬럼 활용)
    if 'LAYER' in df_clean.columns:
        df_clean['LAYER'] = df_clean['LAYER'].astype(int)
    elif 'BUMP_CENTER_Z' in df_clean.columns and df_clean['BUMP_CENTER_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['BUMP_CENTER_Z'].unique())
        z_diffs = np.diff(z_vals)
        gap_threshold = 50.0 
        split_points = z_vals[1:][z_diffs > gap_threshold]
        layers = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layers[df_clean['BUMP_CENTER_Z'] >= p] += 1
        df_clean['LAYER'] = layers
    else:
        df_clean['LAYER'] = 1

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

# 사이드바 설정
st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 10)
p_h = st.sidebar.slider("Plot Height", 3, 15, 8)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")
custom_x_label = st.sidebar.text_input("X-axis Legend", "Shift / Value (um)")
custom_y_label = st.sidebar.text_input("Y-axis Legend", "Layer Number")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale")
v_min = st.sidebar.number_input("Min Limit", value=-10.0)
v_max = st.sidebar.number_input("Max Limit", value=10.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['LAYER'].unique())
        
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Shift Trend"])

        # --- Tab 1: 단일 층 시각화 ---
        with tab1:
            selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['LAYER'] == int(selected_layer.split(" ")[1])]
            
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sc = ax1.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15)
            plt.colorbar(sc, label=f"{d_type} Value")
            ax1.set_title(f"{custom_title} ({selected_layer})")
            ax1.set_xlabel("X (um)"); ax1.set_ylabel("Y (um)")
            if use_custom_scale: sc.set_clim(v_min, v_max)
            st.pyplot(fig1)
            
            # 통계 정보
            stats = display_df.groupby(['SOURCE', 'LAYER'])['Value'].agg(['mean', 'std', 'count']).reset_index()
            st.dataframe(stats)

        # --- Tab 2: Layer별 비교 ---
        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='LAYER', y='Value', hue='SOURCE', ax=ax2)
                ax2.set_title(f"{custom_title}: Layer Comparison")
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                st.pyplot(fig2)
            else:
                st.info("비교 분석을 위해서는 2개 이상의 층이 필요합니다.")

        # --- Tab 3: Multi-Layer Shift Trend (대소문자 해결 버전) ---
        with tab3:
            st.subheader("Multi-Layer Relative Shift Trend")
            
            trend_list = []
            for src in combined_df['SOURCE'].unique():
                src_df = combined_df[combined_df['SOURCE'] == src]
                
                # GROUP_ID가 있는 신규 형식 대응
                if 'GROUP_ID' in src_df.columns and 'X' in src_df.columns:
                    base = src_df[src_df['LAYER'] == 1][['GROUP_ID', 'X', 'Y']]
                    for lyr in sorted(src_df['LAYER'].unique()):
                        target = src_df[src_df['LAYER'] == lyr][['GROUP_ID', 'X', 'Y']]
                        merged = pd.merge(base, target, on='GROUP_ID', suffixes=('_Ref', '_Tgt'))
                        
                        if not merged.empty:
                            # 1층 좌표 대비 상대적 이동량 계산
                            dx = (merged['X_Tgt'] - merged['X_Ref']).mean()
                            dy = (merged['Y_Tgt'] - merged['Y_Ref']).mean()
                            trend_list.append({'SOURCE': src, 'LAYER': lyr, 'DX': dx, 'DY': dy})
            
            if trend_list:
                trend_df = pd.DataFrame(trend_list)
                fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                for src in trend_df['SOURCE'].unique():
                    data = trend_df[trend_df['SOURCE'] == src]
                    ax3.plot(data['DX'], data['LAYER'], marker='o', label=f"{src} (X)")
                    ax3.plot(data['DY'], data['LAYER'], marker='s', ls='--', label=f"{src} (Y)")
                
                ax3.axvline(0, color='black', alpha=0.3)
                ax3.set_yticks(unique_layers)
                ax3.set_xlabel(custom_x_label); ax3.set_ylabel(custom_y_label)
                ax3.set_title(f"{custom_title}: Vertical Shift Trend")
                if use_custom_scale: ax3.set_xlim(v_min, v_max)
                ax3.legend()
                st.pyplot(fig3)
                st.dataframe(trend_df)
                
                csv = trend_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Trend CSV", csv, "Shift_Trend.csv", "text/csv")
            else:
                st.warning("분석에 필요한 컬럼(GROUP_ID, X_COORD 등)을 확인해 주세요.")