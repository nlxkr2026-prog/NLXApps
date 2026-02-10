import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 및 스마트 레이어 분석 ---
def process_data(df, scale_factor, apply_iqr):
    df.columns = [c.strip() for c in df.columns]
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

    # Z-Gap Detection (0, 175, 349... 대형 간격 감지)
    if 'Bump_Center_Z' in df_clean.columns and df_clean['Bump_Center_Z'].nunique() > 1:
        z_vals = np.sort(df_clean['Bump_Center_Z'].unique())
        z_diffs = np.diff(z_vals)
        gap_threshold = 50.0 
        split_points = z_vals[1:][z_diffs > gap_threshold]
        layer_assignment = np.ones(len(df_clean), dtype=int)
        for p in split_points:
            layer_assignment[df_clean['Bump_Center_Z'] >= p] += 1
        df_clean['Layer'] = layer_assignment
    else:
        df_clean['Layer'] = 1
    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Alignment Expert", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Inter-Layer Shift)")

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
custom_x_label = st.sidebar.text_input("X-axis Legend", "Relative Shift (um)")
custom_y_label = st.sidebar.text_input("Y-axis Legend", "Layer Number")

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
        
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Alignment Shift"])

        # --- Tab 1 & 2 (생략: 기존과 동일) ---
        with tab1:
            selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['Layer'] == int(selected_layer.split(" ")[1])]
            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Distribution"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X'], display_df['Y'], c=display_df['Value'], cmap='jet', s=15,
                                vmin=v_min if use_custom_scale else None, vmax=v_max if use_custom_scale else None)
                plt.colorbar(sc, ax=ax1, label=f"{d_type} Value")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='Source', y='Value', ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            elif chart_type == "Distribution":
                sns.histplot(data=display_df, x='Value', hue='Source', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
            ax1.set_title(f"{custom_title} ({selected_layer})")
            st.pyplot(fig1)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='Layer', y='Value', hue='Source', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                ax2.set_title(f"Comparison: {custom_title}")
                st.pyplot(fig2)

        # --- Tab 3: Multi-Layer Alignment Shift (X, Y 위치 직접 계산) ---
        with tab3:
            if len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment Shift (Ref: Layer 1)")
                
                # 1. 층간 매칭을 위한 ID 생성 (좌표 기반)
                combined_df['X_id'] = combined_df['X'].round(1)
                combined_df['Y_id'] = combined_df['Y'].round(1)
                
                # 2. Layer 1을 기준(Reference)으로 설정
                ref_df = combined_df[combined_df['Layer'] == 1][['X_id', 'Y_id', 'X', 'Y', 'Source']]
                
                # 3. 각 층별 상대 Shift 계산
                shift_results = []
                # 기준층(Layer 1)은 Shift 0
                for src in ref_df['Source'].unique():
                    shift_results.append({'Source': src, 'Layer': 1, 'Rel_Shift_X': 0.0, 'Rel_Shift_Y': 0.0})
                
                # 나머지 층 계산
                for layer in unique_layers[1:]:
                    target_df = combined_df[combined_df['Layer'] == layer][['X_id', 'Y_id', 'X', 'Y', 'Source']]
                    merged = pd.merge(ref_df, target_df, on=['X_id', 'Y_id', 'Source'], suffixes=('_Ref', '_Target'))
                    
                    if not merged.empty:
                        merged['DX'] = merged['X_Target'] - merged['X_Ref']
                        merged['DY'] = merged['Y_Target'] - merged['Y_Ref']
                        
                        avg_shifts = merged.groupby('Source')[['DX', 'DY']].mean().reset_index()
                        for _, row in avg_shifts.iterrows():
                            shift_results.append({'Source': row['Source'], 'Layer': layer, 
                                                 'Rel_Shift_X': row['DX'], 'Rel_Shift_Y': row['DY']})
                
                trend_df = pd.DataFrame(shift_results)

                # 4. Vertical Trend 그래프 그리기
                fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                for src in trend_df['Source'].unique():
                    src_data = trend_df[trend_df['Source'] == src]
                    ax3.plot(src_data['Rel_Shift_X'], src_data['Layer'], marker='o', label=f"{src} (X Rel)")
                    ax3.plot(src_data['Rel_Shift_Y'], src_data['Layer'], marker='s', linestyle='--', label=f"{src} (Y Rel)")
                
                ax3.axvline(0, color='black', lw=1, alpha=0.3)
                ax3.set_yticks(unique_layers)
                ax3.set_ylabel(custom_y_label)
                ax3.set_xlabel(custom_x_label)
                ax3.set_title(f"{custom_title}: Alignment Trend (Relative to Layer 1)")
                ax3.legend()
                
                if use_custom_scale: ax3.set_xlim(v_min, v_max)
                st.pyplot(fig3)
                
                st.write("**Calculated Relative Shift Data**")
                st.dataframe(trend_df)
                st.download_button("📥 Export Alignment CSV", trend_df.to_csv(index=False).encode('utf-8'), "Alignment_Trend.csv", "text/csv")
            else:
                st.info("2개 이상의 층이 있어야 상대적인 Shift 분석이 가능합니다.")