import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 및 스마트 레이어 분석 로직 (Z-Gap Detection) ---
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
    
    # 3. 이상치 제거 (IQR 체크박스 연동)
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # 4. 스마트 레이어 분석 (Z-Gap Detection: um 단위 대형 간격 감지)
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['Bump_Center_Z'].unique())
        z_diffs = np.diff(z_vals)
        z_range = z_vals.max() - z_vals.min()
        gap_threshold = max(z_range * 0.1, 50.0) 
        
        split_points = z_vals[1:][z_diffs > gap_threshold]
        layer_assignment = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layer_assignment[df_clean['Bump_Center_Z'] >= p] += 1
        df_clean['Layer'] = layer_assignment
    else:
        df_clean['Layer'] = 1

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Expert", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Multi-Layer & Export)")

# 사이드바 설정
st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True) # IQR 기능 복구

# 그래프 크기 조절
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 12)
p_h = st.sidebar.slider("Plot Height", 3, 15, 6)

# 그래프 커스터마이징 (전체 적용) - 복구 완료
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Analysis Result")
custom_x_label = st.sidebar.text_input("X-axis Legend", "X Position (um)")
custom_y_label = st.sidebar.text_input("Y-axis Legend", "Y Position (um)")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Value Scale")
v_min = st.sidebar.number_input("Value Min", value=0.0)
v_max = st.sidebar.number_input("Value Max", value=20.0)

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
        
        # 상단 탭 구성
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "🔄 Multi-Layer Shift"])

        # --- Tab 1: 단일 층 분석 ---
        with tab1:
            selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]
            
            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Distribution", "X-Y Shift Scatter"], horizontal=True)
            fig, ax = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                vm_min = v_min if use_custom_scale else display_df['Value'].min()
                vm_max = v_max if use_custom_scale else display_df['Value'].max()
                sc = ax.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15, vmin=vm_min, vmax=vm_max)
                plt.colorbar(sc, label=f"{d_type} Value")
                ax.set_xlabel(custom_x_label); ax.set_ylabel(custom_y_label)

            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='Source', y='Value', ax=ax)
                ax.set_xlabel("Source Files"); ax.set_ylabel(f"{d_type} Value")
                if use_custom_scale: ax.set_ylim(v_min, v_max)
                plt.xticks(rotation=45)

            elif chart_type == "Distribution":
                sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax)
                ax.set_xlabel(f"{d_type} Value")
                if use_custom_scale: ax.set_xlim(v_min, v_max)

            elif chart_type == "X-Y Shift Scatter":
                for src in display_df['Source'].unique():
                    src_df = display_df[display_df['Source'] == src]
                    if 'Shift_X' in src_df.columns:
                        ax.scatter(src_df['Shift_X']*scale, src_df['Shift_Y']*scale, label=src, alpha=0.6, s=15)
                ax.axhline(0, color='black', lw=1); ax.axvline(0, color='black', lw=1)
                ax.set_xlabel(f"Shift X ({custom_x_label})"); ax.set_ylabel(f"Shift Y ({custom_y_label})")
                if use_custom_scale: ax.set_xlim(v_min, v_max); ax.set_ylim(v_min, v_max)
                ax.legend()
            
            ax.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig)

            # 통계 데이터 복구 및 Export 기능 추가
            st.markdown("---")
            st.subheader(f"📋 Summary Statistics ({selected_layer})")
            summary_df = display_df.groupby(['Source', 'Layer'])['Value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            summary_df['3-Sigma'] = summary_df['std'] * 3
            st.dataframe(summary_df, use_container_width=True)

            # Export 기능
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Statistics as CSV", data=csv, file_name=f"NLX_Stats_{selected_layer}.csv", mime='text/csv')

        # --- Tab 2: 층별 비교 (Boxplot) ---
        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='Layer', y='Value', hue='Source', ax=ax2)
                ax2.set_title(f"{custom_title} - Layer-wise Comparison")
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                st.pyplot(fig2)
            else: st.info("Multi-layer data is required for comparison.")

        # --- Tab 3: Multi-Layer Shift (Ref: Layer 1) ---
        with tab3:
            if len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Shift (Reference: Layer 1)")
                combined_df['X_id'], combined_df['Y_id'] = combined_df['X'].round(1), combined_df['Y'].round(1)
                base = combined_df[combined_df['Layer'] == 1][['X_id', 'Y_id', 'X', 'Y', 'Source']]
                targets = combined_df[combined_df['Layer'] > 1]
                merged = pd.merge(base, targets, on=['X_id', 'Y_id', 'Source'], suffixes=('_L1', '_LN'))
                merged['DX'], merged['DY'] = merged['X_LN'] - merged['X_L1'], merged['Y_LN'] - merged['Y_L1']
                merged['Total_Shift'] = np.sqrt(merged['DX']**2 + merged['DY']**2)
                
                fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                sns.scatterplot(data=merged, x='DX', y='DY', hue='Layer', ax=ax3, alpha=0.7)
                ax3.set_title(f"{custom_title} - Alignment Shift")
                ax3.set_xlabel(f"Delta X ({custom_x_label})"); ax3.set_ylabel(f"Delta Y ({custom_y_label})")
                st.pyplot(fig3)
                
                shift_stats = merged.groupby(['Source', 'Layer'])['Total_Shift'].describe().reset_index()
                st.write("**Shift Statistics (um)**")
                st.dataframe(shift_stats)
                
                # Shift 데이터도 Export 가능
                csv_shift = shift_stats.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Shift Stats as CSV", data=csv_shift, file_name="NLX_MultiLayer_Shift_Stats.csv", mime='text/csv')