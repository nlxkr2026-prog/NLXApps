import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- [1] 데이터 전처리 로직 (Z-Gap 0번부터 할당 및 대소문자 표준화) ---
def process_data(df, scale_factor, apply_iqr):
    # 컬럼명 대문자 표준화
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. 데이터 타입 판별 및 타겟 설정
    d_type = None
    target_cols = []
    
    if 'HEIGHT' in df.columns: 
        d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns: 
        d_type, target_cols = "Radius", ['RADIUS']
    elif 'SHIFT_NORM' in df.columns or 'SHIFT_X' in df.columns: 
        d_type = "Shift"
        if 'SHIFT_NORM' in df.columns: target_cols.append('SHIFT_NORM')
        if 'SHIFT_X' in df.columns: target_cols.append('SHIFT_X')
        if 'SHIFT_Y' in df.columns: target_cols.append('SHIFT_Y')
    elif 'X_COORD' in df.columns: 
        d_type, target_cols = "Coordinate", ['X_COORD']
    else: return None, None

    # 2. 좌표 및 측정값 설정
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    
    # 모든 가용 측정 컬럼에 스케일 적용
    for col in target_cols:
        df[col + '_UM'] = df[col] * scale_factor
    
    # 기본 메인 값 설정
    df['MEAS_VALUE'] = df[target_cols[0] + '_UM']
    
    # 3. 레이어 번호 설정 (0번부터 시작)
    if 'LAYER_NUMBER' in df.columns:
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        if len(z_vals) > 1:
            z_diffs = np.diff(z_vals)
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            l_assign = np.zeros(len(df), dtype=int)
            for p in splits: l_assign[df['BUMP_CENTER_Z'] >= p] += 1
            df['L_NUM'] = l_assign
        else:
            df['L_NUM'] = 0
    else:
        df['L_NUM'] = 0

    # 4. Pillar 식별자
    df['P_ID'] = df['PILLAR_NUMBER'] if 'PILLAR_NUMBER' in df.columns else (df['GROUP_ID'] if 'GROUP_ID' in df.columns else None)

    # 5. IQR 필터링
    df_clean = df.copy()
    if apply_iqr and d_type != "Coordinate":
        df_clean = df_clean[df_clean['MEAS_VALUE'] != 0]
        if not df_clean.empty:
            q1, q3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_clean = df_clean[(df_clean['MEAS_VALUE'] >= q1 - 1.5 * iqr) & (df_clean['MEAS_VALUE'] <= q3 + 1.5 * iqr)]

    return df_clean, d_type

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Multi-Layer Professional", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Ref: Layer 0)")

st.sidebar.header("📁 Configuration")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)
use_iqr = st.sidebar.checkbox("Apply IQR Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Plot Size Settings")
p_w = st.sidebar.slider("Plot Width", 5, 25, 10)
p_h = st.sidebar.slider("Plot Height", 3, 15, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Plot Customization")
custom_title = st.sidebar.text_input("Graph Title", "Alignment Analysis")
custom_x_legend = st.sidebar.text_input("X-axis Legend Name", "X Position (um)")
custom_y_legend = st.sidebar.text_input("Y-axis Legend Name", "Y Position (um)")

st.sidebar.subheader("📏 Scale Settings")
use_custom_scale = st.sidebar.checkbox("Apply Custom Scale Range")
v_min = st.sidebar.number_input("Min Limit", value=-10.0)
v_max = st.sidebar.number_input("Max Limit", value=10.0)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_iqr)
        if p_df is not None:
            p_df['SOURCE_FILE'] = file.name
            all_data.append(p_df)

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())
        
        tab1, tab2, tab3 = st.tabs(["📊 Single Layer View", "📈 Layer Comparison", "📉 Multi-Layer Shift Trend"])

        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                selected_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            
            display_df = combined_df if selected_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(selected_layer.split(" ")[1])]
            
            active_target = 'MEAS_VALUE'
            if d_type == "Shift":
                avail_cols = [c for c in display_df.columns if c.endswith('_UM')]
                with c2:
                    active_target = st.selectbox("Select Target Data", avail_cols, index=0)

            chart_type = st.radio("Chart Type", ["Heatmap", "Box Plot", "Histogram"], horizontal=True)
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            
            if chart_type == "Heatmap":
                sc = ax1.scatter(display_df['X_VAL'], display_df['Y_VAL'], c=display_df[active_target], cmap='jet', s=15)
                if use_custom_scale: sc.set_clim(v_min, v_max)
                plt.colorbar(sc, label=f"Value (um)")
            elif chart_type == "Box Plot":
                sns.boxplot(data=display_df, x='SOURCE_FILE', y=active_target, ax=ax1)
                if use_custom_scale: ax1.set_ylim(v_min, v_max)
            elif chart_type == "Histogram":
                sns.histplot(data=display_df, x=active_target, hue='SOURCE_FILE', kde=True, ax=ax1)
                if use_custom_scale: ax1.set_xlim(v_min, v_max)
            
            ax1.set_title(f"{custom_title} ({selected_layer}) - {active_target}")
            ax1.set_xlabel(custom_x_legend); ax1.set_ylabel(custom_y_legend)
            st.pyplot(fig1)

            # [탭1 통계 추가]
            st.markdown("---")
            st.subheader(f"📊 Single Layer Statistics ({selected_layer})")
            s_stats = display_df.groupby('SOURCE_FILE')[active_target].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            s_stats['3-Sigma'] = s_stats['std'] * 3
            st.dataframe(s_stats)

        with tab2:
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                ax2.set_title(f"Layer Comparison: {custom_title}")
                ax2.set_xlabel("Layer Number"); ax2.set_ylabel(f"{d_type} Value")
                st.pyplot(fig2)

                # [탭2 통계 추가]
                st.markdown("---")
                st.subheader("📊 Layer-wise Comparison Statistics")
                c_stats = combined_df.groupby(['SOURCE_FILE', 'L_NUM'])['MEAS_VALUE'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                c_stats['3-Sigma'] = c_stats['std'] * 3
                st.dataframe(c_stats)
                st.download_button("📥 Export Comparison Stats CSV", c_stats.to_csv(index=False).encode('utf-8'), "layer_stats.csv")
            else:
                st.info("비교를 위해서는 2층 이상의 레이어가 필요합니다.")

        with tab3:
            shift_df = combined_df.dropna(subset=['P_ID'])
            if not shift_df.empty and len(unique_layers) > 1:
                st.subheader("Inter-Layer Alignment & Shift Magnitude Heatmap")
                
                trend_list = []
                heatmap_data_list = []
                
                for src in shift_df['SOURCE_FILE'].unique():
                    src_df = shift_df[shift_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    
                    if not base.empty:
                        for lyr in unique_layers:
                            target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                            merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                            if not merged.empty:
                                merged['DX'] = merged['X_VAL_TGT'] - merged['X_VAL_REF']
                                merged['DY'] = merged['Y_VAL_TGT'] - merged['Y_VAL_REF']
                                merged['MAG'] = np.sqrt(merged['DX']**2 + merged['DY']**2)
                                
                                trend_list.append({'Source': src, 'Layer': lyr, 'Avg_DX': merged['DX'].mean(), 'Avg_DY': merged['DY'].mean(), 'Avg_Mag': merged['MAG'].mean()})
                                merged['Layer'] = lyr; merged['Source'] = src
                                heatmap_data_list.append(merged)

                if trend_list:
                    trend_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for src in trend_df['Source'].unique():
                        data = trend_df[trend_df['Source'] == src]
                        ax3.plot(data['Avg_DX'], data['Layer'], marker='o', label=f"{src} (X Avg)")
                        ax3.plot(data['Avg_DY'], data['Layer'], marker='s', ls='--', label=f"{src} (Y Avg)")
                    ax3.axvline(0, color='black', alpha=0.3)
                    ax3.set_yticks(unique_layers)
                    ax3.set_xlabel("Average Shift (um)"); ax3.set_ylabel("Layer Number")
                    ax3.set_title(f"{custom_title}: Vertical Shift Trend")
                    ax3.legend()
                    st.pyplot(fig3)

                    # [탭3 통계 추가]
                    st.markdown("---")
                    st.subheader("📊 Multi-Layer Shift Trend Statistics")
                    st.dataframe(trend_df)
                    st.download_button("📥 Export Trend CSV", trend_df.to_csv(index=False).encode('utf-8'), "alignment_trend.csv")
                    
                    st.markdown("---")
                    st.subheader("Shift Magnitude Map (Relative to Layer 0)")
                    h_layer = st.selectbox("Select Layer for Heatmap", unique_layers[1:])
                    h_df_all = pd.concat(heatmap_data_list)
                    h_df = h_df_all[h_df_all['Layer'] == h_layer]
                    
                    fig4, ax4 = plt.subplots(figsize=(p_w, p_h))
                    sc_h = ax4.scatter(h_df['X_VAL_REF'], h_df['Y_VAL_REF'], c=h_df['MAG'], cmap='Reds', s=20)
                    plt.colorbar(sc_h, label="Shift Magnitude (um)")
                    ax4.set_title(f"Layer {h_layer} Shift Intensity Map")
                    ax4.set_xlabel(custom_x_legend); ax4.set_ylabel(custom_y_legend)
                    st.pyplot(fig4)
            else:
                st.info("Pillar/Coordinate 데이터와 Layer 0가 필요합니다.")