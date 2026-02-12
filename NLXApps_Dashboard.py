import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import os

# --- [1] 데이터 전처리 및 정밀 분석 로직 ---
def process_data(df, scale_factor, apply_iqr, pitch_sensitivity):
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 데이터 성격 판별 플래그
    is_shift_data = 'X_COORD' in df.columns
    d_type = None
    target_cols = []
    
    # 1. 데이터 타입 판별 및 타겟 설정
    if is_shift_data:
        d_type = "Shift"
        for c in ['SHIFT_NORM', 'SHIFT_X', 'SHIFT_Y', 'X_COORD', 'Y_COORD']:
            if c in df.columns: target_cols.append(c)
    elif 'HEIGHT' in df.columns: 
        d_type, target_cols = "Height", ['HEIGHT']
    elif 'RADIUS' in df.columns: 
        d_type, target_cols = "Radius", ['RADIUS']
    else: 
        return None, None, False

    # 2. 좌표 설정 (정수형 scale_factor 적용)
    df['X_VAL'] = (df['X_COORD'] if 'X_COORD' in df.columns else df.get('BUMP_CENTER_X', 0)) * scale_factor
    df['Y_VAL'] = (df['Y_COORD'] if 'Y_COORD' in df.columns else df.get('BUMP_CENTER_Y', 0)) * scale_factor
    
    for col in target_cols:
        df[col + '_UM'] = df[col] * scale_factor
    
    # 메인 측정값 설정
    main_col = 'SHIFT_NORM' if 'SHIFT_NORM' in df.columns else target_cols[0]
    df['MEAS_VALUE'] = df[main_col + '_UM']

    # 3. 레이어 설정
    if 'LAYER_NUMBER' in df.columns: 
        df['L_NUM'] = df['LAYER_NUMBER'].astype(int)
    elif 'BUMP_CENTER_Z' in df.columns:
        # Z값을 이용한 자동 레이어 분리 로직 유지
        z_vals = np.sort(df['BUMP_CENTER_Z'].unique())
        if len(z_vals) > 1:
            z_diffs = np.diff(z_vals)
            gap = max((z_vals.max() - z_vals.min()) * 0.1, 0.05)
            splits = z_vals[1:][z_diffs > gap]
            l_assign = np.zeros(len(df), dtype=int)
            for p in splits: l_assign[df['BUMP_CENTER_Z'] >= p] += 1
            df['L_NUM'] = l_assign
        else: df['L_NUM'] = 0
    else: df['L_NUM'] = 0

    # 4. 고유 식별자(P_ID) 설정
    if 'PILLAR_NUMBER' in df.columns: df['P_ID'] = df['PILLAR_NUMBER']
    elif 'GROUP_ID' in df.columns: df['P_ID'] = df['GROUP_ID']
    else: df['P_ID'] = df.index

    # 5. [개선] Pitch 알고리즘 (ID 연속성 및 평균 1.5배 필터 반영)
    grid_size = 0.5
    group_base = ['SOURCE_FILE', 'L_NUM']
    
    # X-Pitch: 동일 행(Y_GRID) 내 계산
    df['Y_GRID'] = (df['Y_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['Y_GRID', 'X_VAL'])
    df['ID_DIFF'] = df.groupby(group_base + ['Y_GRID'])['P_ID'].diff()
    # ID가 연속적일 때만 X-Pitch 인정
    df['X_P_RAW'] = np.where(df['ID_DIFF'] == 1, df.groupby(group_base + ['Y_GRID'])['X_VAL'].diff().abs(), np.nan)

    # Y-Pitch: 동일 열(X_GRID) 내 계산
    df['X_GRID'] = (df['X_VAL'] / grid_size).round() * grid_size
    df = df.sort_values(by=group_base + ['X_GRID', 'Y_VAL'])
    df['Y_P_RAW'] = df.groupby(group_base + ['X_GRID'])['Y_VAL'].diff().abs()

    # 평균 대비 1.5배 필터 (Missing Bump에 의한 배수 Pitch 제거)
    for col in ['X_P_RAW', 'Y_P_RAW']:
        valid_data = df[col].dropna()
        if not valid_data.empty:
            avg_p = valid_data.mean()
            df[col] = np.where((df[col] > avg_p * 1.5) | (df[col] < avg_p * 0.5), np.nan, df[col])
    
    # Pitch 전용 가변 IQR 필터 적용
    for col in ['X_P_RAW', 'Y_P_RAW']:
        valid_p = df[col].dropna()
        if not valid_p.empty:
            q1, q3 = valid_p.quantile([0.25, 0.75])
            iqr_p = q3 - q1
            df.loc[(df[col] < q1 - pitch_sensitivity*iqr_p) | (df[col] > q3 + pitch_sensitivity*iqr_p), col] = np.nan

    df['X_PITCH'] = df['X_P_RAW']
    df['Y_PITCH'] = df['Y_P_RAW']

    # 6. 측정값 IQR 필터
    df_clean = df[df['MEAS_VALUE'] != 0].copy()
    if apply_iqr and d_type != "Coordinate":
        qh1, qh3 = df_clean['MEAS_VALUE'].quantile([0.25, 0.75])
        iqr_h = qh3 - qh1
        df_clean = df_clean[(df_clean['MEAS_VALUE'] >= qh1 - 1.5*iqr_h) & (df_clean['MEAS_VALUE'] <= qh3 + 1.5*iqr_h)]

    return df_clean, d_type, is_shift_data

# --- 범례 관리 함수 ---
def apply_global_legend(ax, loc, show_legend):
    if not show_legend:
        leg = ax.get_legend()
        if leg: leg.remove()
        return
    try:
        sns.move_legend(ax, loc=loc, title=None)
    except:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles=handles, labels=labels, loc=loc, title=None)

# --- [2] UI 구성 ---
st.set_page_config(page_title="NLX Professional Analysis", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard")

with st.sidebar:
    st.header("📁 Data Config")
    uploaded_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
    
    # [수정] Scale Factor 정수만 입력 가능하도록 설정
    scale = st.number_input("Multiplier (Integer only)", value=1, step=1, format="%d", help="정수 배율만 입력 가능합니다. (예: 1000)")
    use_iqr = st.checkbox("Apply IQR Filter (Meas. Value)", value=True)
    
    st.markdown("---")
    st.subheader("🎯 Pitch Filter Settings")
    pitch_sensitivity = st.slider("Pitch Outlier Sensitivity", 0.5, 3.0, 1.2, 0.1)

    with st.expander("🎨 Plot Settings", expanded=True):
        p_w = st.slider("Width", 5, 25, 12)
        p_h = st.slider("Height", 3, 15, 6)
        custom_title = st.text_input("Title", "Alignment Analysis")

    with st.expander("📏 Scale Control", expanded=False):
        show_legend = st.checkbox("Show Legend", value=True)
        global_legend_loc = st.selectbox("Legend Loc", options=["best", "upper right", "right", "center"], index=1)
        use_custom_scale = st.checkbox("Manual Axis Range", value=False)
        v_min = st.number_input("Min", value=-10.0)
        v_max = st.number_input("Max", value=10.0)

    with st.expander("🧊 3D View Settings", expanded=False):
        color_option = st.selectbox("Color Theme", ["Viridis", "Plasma", "Jet", "Turbo"])
        use_outlier_filter = st.checkbox("Highlight Outliers in 3D")
        outlier_low = st.number_input("3D Lower Bound (Yellow)", value=-5.0)
        outlier_high = st.number_input("3D Upper Bound (Red)", value=5.0)

if uploaded_files:
    all_data = []
    has_any_shift_data = False
    
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        # scale 인자는 정수로 전달됨
        p_df, d_type, is_shift = process_data(raw_df, int(scale), use_iqr, pitch_sensitivity)
        if p_df is not None:
            p_df['SOURCE_FILE'] = os.path.splitext(file.name)[0]
            all_data.append(p_df)
            if is_shift: has_any_shift_data = True

    if all_data:
        combined_df = pd.concat(all_data)
        unique_layers = sorted(combined_df['L_NUM'].unique())

        st.markdown("### 📋 Quick Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global Average", f"{combined_df['MEAS_VALUE'].mean():.3f}")
        m2.metric("Global 3-Sigma", f"{(combined_df['MEAS_VALUE'].std()*3):.3f}")
        m3.metric("Max-Min Range", f"{(combined_df['MEAS_VALUE'].max()-combined_df['MEAS_VALUE'].min()):.3f}")
        m4.metric("Total Bumps", f"{len(combined_df):,}")

        tabs = st.tabs(["📊 Single Layer", "📈 Comparison", "📉 Shift Trend", "🧊 3D View", "🎯 Pitch Analysis"])

        with tabs[0]: # Single Layer
            sel_layer = st.selectbox("Select Layer", ["All Layers"] + [f"Layer {i}" for i in unique_layers])
            disp_df = combined_df if sel_layer == "All Layers" else combined_df[combined_df['L_NUM'] == int(sel_layer.split(" ")[1])]
            fig1, ax1 = plt.subplots(figsize=(p_w, p_h))
            sns.histplot(data=disp_df, x='MEAS_VALUE', hue='SOURCE_FILE', kde=True, ax=ax1)
            if use_custom_scale: ax1.set_xlim(v_min, v_max)
            apply_global_legend(ax1, global_legend_loc, show_legend)
            st.pyplot(fig1)

        with tabs[1]: # Comparison
            if len(unique_layers) > 1:
                fig2, ax2 = plt.subplots(figsize=(p_w, p_h))
                sns.boxplot(data=combined_df, x='L_NUM', y='MEAS_VALUE', hue='SOURCE_FILE', ax=ax2)
                if use_custom_scale: ax2.set_ylim(v_min, v_max)
                apply_global_legend(ax2, global_legend_loc, show_legend)
                st.pyplot(fig2)
            else: st.info("Analysis requires more than one layer.")

        with tabs[2]: # Shift Trend
            if not has_any_shift_data:
                st.warning("⚠️ Shift Trend 분석은 'X_COORD' 데이터가 포함된 전용 파일에서만 활성화됩니다.")
            else:
                trend_list = []
                for src in combined_df['SOURCE_FILE'].unique():
                    src_df = combined_df[combined_df['SOURCE_FILE'] == src]
                    base = src_df[src_df['L_NUM'] == 0][['P_ID', 'X_VAL', 'Y_VAL']]
                    if base.empty: continue
                    for lyr in unique_layers:
                        if lyr == 0: continue
                        target = src_df[src_df['L_NUM'] == lyr][['P_ID', 'X_VAL', 'Y_VAL']]
                        merged = pd.merge(base, target, on='P_ID', suffixes=('_REF', '_TGT'))
                        if not merged.empty:
                            trend_list.append({'Source': src, 'Layer': lyr, 
                                               'Avg_DX': (merged['X_VAL_TGT'] - merged['X_VAL_REF']).mean(), 
                                               'Avg_DY': (merged['Y_VAL_TGT'] - merged['Y_VAL_REF']).mean()})
                if trend_list:
                    t_df = pd.DataFrame(trend_list)
                    fig3, ax3 = plt.subplots(figsize=(p_w, p_h))
                    for s in t_df['Source'].unique():
                        d = t_df[t_df['Source'] == s].sort_values('Layer')
                        ax3.plot(d['Avg_DX'], d['Layer'], marker='o', label=f"{s} (DX)")
                        ax3.plot(d['Avg_DY'], d['Layer'], marker='s', ls='--', label=f"{s} (DY)")
                    if use_custom_scale: ax3.set_xlim(v_min, v_max)
                    ax3.set_title("Shift Trend from Layer 0"); ax3.legend(); st.pyplot(fig3)

        with tabs[3]: # 3D View
            plot_3d_df = combined_df.copy()
            if use_outlier_filter:
                cond = [(plot_3d_df['MEAS_VALUE'] < outlier_low), (plot_3d_df['MEAS_VALUE'] > outlier_high)]
                plot_3d_df['Status'] = np.select(cond, ['Under (Yellow)', 'Over (Red)'], default='Normal')
                fig3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='Status',
                                     color_discrete_map={'Under (Yellow)': 'yellow', 'Over (Red)': 'red', 'Normal': 'blue'})
            else:
                fig3d = px.scatter_3d(plot_3d_df, x='X_VAL', y='Y_VAL', z='L_NUM', color='MEAS_VALUE', color_continuous_scale=color_option.lower())
            fig3d.update_layout(height=700); st.plotly_chart(fig3d, use_container_width=True)

        with tabs[4]: # Pitch Analysis
            st.subheader("🎯 Pitch Analysis")
            sel_layer_p = st.selectbox("Select Layer for Pitch", ["All Layers"] + [f"Layer {i}" for i in unique_layers], key="p_sel")
            p_df = combined_df if sel_layer_p == "All Layers" else combined_df[combined_df['L_NUM'] == int(sel_layer_p.split(" ")[1])]
            
            c1, c2 = st.columns(2)
            for col, p_type, p_color in zip([c1, c2], ['X_PITCH', 'Y_PITCH'], ['Blues', 'Reds']):
                with col:
                    fig_p, ax_p = plt.subplots(figsize=(p_w/2, p_h))
                    sns.boxplot(data=p_df, x='SOURCE_FILE', y=p_type, hue='SOURCE_FILE', ax=ax_p, palette=p_color)
                    apply_global_legend(ax_p, global_legend_loc, show_legend); st.pyplot(fig_p)
            
            st.dataframe(p_df.groupby('SOURCE_FILE')[['X_PITCH', 'Y_PITCH']].mean().style.format("{:.3f}"), use_container_width=True)
else:
    st.info("Please upload CSV files.")