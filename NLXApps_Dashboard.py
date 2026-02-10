import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 및 스마트 레이어 분석 로직 ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # IQR 필터링
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 스마트 레이어 분석 (Z-Gap Detection)
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['Bump_Center_Z'].unique())
        z_diffs = np.diff(z_vals)
        gap_threshold = 50.0 # um 단위 대응
        split_points = z_vals[1:][z_diffs > gap_threshold]
        
        layer_assignment = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layer_assignment[df_clean['Bump_Center_Z'] >= p] += 1
        df_clean['Layer'] = layer_assignment
    else:
        df_clean['Layer'] = 1

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Professional Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Final Version)")

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
custom_y_label = st.sidebar.text_input("Y-axis Legend", "Y Position / Layer")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Value Scale")
v_min = st.sidebar.number_input("Value Min", value=-10.0)
v_max = st.sidebar.number_input("Value Max", value=10.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['Layer'].unique())
        
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Shift Trend"])

        with tab1:
            selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]
            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15,
                                vmin=v_min if use_custom_scale else None, vmax=v_max if use_custom_scale else None)
                plt.colorbar(sc, ax=ax1, label=f"{d_type} Value")
                ax1.set_xlabel(custom_x_label); ax1.set_ylabel(custom_y_label)
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='Source', y='Value', ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            elif chart_type == "Distribution":
                sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
            ax1.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig1)
            
            # 통계 데이터 Export
            summary_df = display_df.groupby(['Source', 'Layer'])['Value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            summary_df['3-Sigma'] = summary_df['std'] * 3
            st.dataframe(summary_df)
            st.download_button("📥 Export Stats CSV", summary_df.to_csv(index=False).encode('utf-8'), "Stats.csv", "text/csv")

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='Layer', y='Value', hue='Source', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                ax2.set_title(f"Comparison: {custom_title}")
                st.pyplot(fig2)

        with tab3:
            if len(unique_layers) > 1 and ('Shift_X' in combined_df.columns):
                st.subheader("Vertical Shift Trend (Y: Layer, X: Average Shift)")
                trend_df = combined_df.groupby(['Source', 'Layer'])[['Shift_X', 'Shift_Y']].mean().reset_index()
                trend_df['Shift_X_um'] = trend_df['Shift_X'] * scale
                trend_df['Shift_Y_um'] = trend_df['Shift_Y'] * scale

                fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                for src in trend_df['Source'].unique():
                    src_data = trend_df[trend_df['Source'] == src]
                    ax3.plot(src_data['Shift_X_um'], src_data['Layer'], marker='o', label=f"{src} (X Avg)")
                    ax3.plot(src_data['Shift_Y_um'], src_data['Layer'], marker='s', linestyle='--', label=f"{src} (Y Avg)")
                
                ax3.axvline(0, color='black', lw=1, alpha=0.3)
                ax3.set_yticks(unique_layers)
                ax3.set_ylabel("Layer Number")
                ax3.set_xlabel(custom_x_label)
                ax3.set_title(f"{custom_title}: Vertical Shift Trend")
                ax3.legend()
                if use_custom_scale: ax3.set_xlim(v_min, v_max)
                st.pyplot(fig3)
                st.download_button("📥 Export Trend CSV", trend_df.to_csv(index=False).encode('utf-8'), "Trend.csv", "text/csv")
            else:
                st.info("Shift 데이터가 포함된 2층 이상의 데이터가 필요합니다.")