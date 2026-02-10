import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
<<<<<<< HEAD
from sklearn.cluster import KMeans  # 레이어 분리를 위해 추가

# --- [1] 데이터 전처리 및 레이어 분석 로직 ---
=======

# --- [1] 데이터 전처리 및 Pitch 계산 로직 (IQR 필터링 고도화) ---
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
def process_data(df, scale_factor, apply_iqr, apply_pitch_iqr):
    df.columns = [c.strip() for c in df.columns]
    
    # 데이터 타입 판별
    if 'Height' in df.columns: d_type, target = "Height", "Height"
    elif 'Radius' in df.columns: d_type, target = "Radius", "Radius"
    elif 'Shift_Norm' in df.columns: d_type, target = "Shift", "Shift_Norm"
    else: return None, None

<<<<<<< HEAD
    # [추가] 레이어 자동 분석 (Z-Position 기반 클러스터링)
    # Z값의 차이가 미세하므로 클러스터링을 통해 층을 구분합니다.
    z_values = df['Bump_Center_Z'].values.reshape(-1, 1)
    
    # 엘보우 포인트 대신 최대 5개 층까지 탐색하여 최적의 층 수 계산 (간단한 로직)
    # 실무적으로는 사용자가 층 수를 입력하게 할 수도 있습니다.
    n_clusters = 1
    if len(df) > 10:
        # Z값의 고유값 범위를 보고 대략적인 층수 추정 (차이가 0.005 이상일 때 구분 등)
        z_range = np.ptp(df['Bump_Center_Z'])
        if z_range > 0.01: n_clusters = 2 # 예시 임계치
        if z_range > 0.05: n_clusters = 3
    
    # 사이드바에서 선택할 수 있도록 일단 1~5층 사이에서 자동 할당하거나 
    # 아래 메인 루프에서 사용자가 지정한 n_layers를 사용할 수 있습니다.
    
    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Z_um'] = df['Bump_Center_Z'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 1차: 메인 Value IQR 제거
=======
    # 기본 단위 변환
    df['X'] = df['Bump_Center_X'] * scale_factor
    df['Y'] = df['Bump_Center_Y'] * scale_factor
    df['Value'] = df[target] * scale_factor
    
    # 1차: 메인 Value(Height/Radius/Shift) IQR 제거
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
    df_clean = df[df['Value'] != 0].copy()
    if apply_iqr:
        q1, q3 = df_clean['Value'].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_clean = df_clean[(df_clean['Value'] >= q1 - 1.5 * iqr) & (df_clean['Value'] <= q3 + 1.5 * iqr)]

<<<<<<< HEAD
    # 2차: Pitch 계산
=======
    # 2차: Pitch 계산 (그리드 기반)
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
    df_clean['Y_grid'] = df_clean['Y'].round(0)
    df_clean = df_clean.sort_values(by=['Y_grid', 'X'])
    df_clean['X_Pitch'] = df_clean.groupby('Y_grid')['X'].diff()

    df_clean['X_grid'] = df_clean['X'].round(0)
    df_clean = df_clean.sort_values(by=['X_grid', 'Y'])
    df_clean['Y_Pitch'] = df_clean.groupby('X_grid')['Y'].diff()

<<<<<<< HEAD
    # 3차: Pitch IQR 필터링
=======
    # 3차: [추가] Pitch 데이터 IQR 필터링 (선택 사항)
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
    if apply_pitch_iqr:
        for col in ['X_Pitch', 'Y_Pitch']:
            p_data = df_clean[col].dropna()
            if not p_data.empty:
                pq1, pq3 = p_data.quantile([0.25, 0.75])
                piqr = pq3 - pq1
<<<<<<< HEAD
=======
                # 이상치에 해당하는 행의 Pitch 값만 NaN으로 처리하여 통계/그래프에서 제외
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
                df_clean.loc[(df_clean[col] < pq1 - 1.5 * piqr) | (df_clean[col] > pq3 + 1.5 * piqr), col] = np.nan

    return df_clean, d_type

# --- [2] UI 구성 ---
<<<<<<< HEAD
st.set_page_config(page_title="NLX Multi-Layer Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (Layer Analysis)")

st.sidebar.header("📁 Data & Layer Settings")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor", value=1000)

# [추가] 레이어 분리 설정
n_layers = st.sidebar.slider("Number of expected layers (Z-axis)", 1, 5, 1)

st.sidebar.subheader("🛡️ Outlier Removal Settings")
use_val_iqr = st.sidebar.checkbox("Apply IQR to Value", value=True)
use_pitch_iqr = st.sidebar.checkbox("Apply IQR to Pitch", value=True)

if uploaded_files:
    all_data = []
    
    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        p_df, d_type = process_data(raw_df, scale, use_val_iqr, use_pitch_iqr)
        
        if p_df is not None:
            # Z축 클러스터링 수행 (레이어 할당)
            if n_layers > 1:
                kmeans = KMeans(n_clusters=n_layers, random_state=42)
                p_df['Layer'] = kmeans.fit_predict(p_df[['Bump_Center_Z']])
                # Z값 평균 순서대로 레이어 이름 재정렬 (0층이 가장 낮은 층이 되도록)
                layer_order = p_df.groupby('Layer')['Bump_Center_Z'].mean().sort_values().index
                layer_map = {old: new for new, old in enumerate(layer_order)}
                p_df['Layer'] = p_df['Layer'].map(layer_map)
            else:
                p_df['Layer'] = 0
                
            p_df['Source'] = file.name
            all_data.append(p_df)

    combined_df = pd.concat(all_data)

    # 레이어 필터링 UI
    st.sidebar.markdown("---")
    unique_layers = sorted(combined_df['Layer'].unique())
    selected_layer = st.sidebar.selectbox("Select Layer to View", ["All Layers"] + [f"Layer {i}" for i in unique_layers])

    # 데이터 필터링 실행
    if selected_layer != "All Layers":
        layer_num = int(selected_layer.split(" ")[1])
        display_df = combined_df[combined_df['Layer'] == layer_num]
    else:
        display_df = combined_df

    # 상단 요약 요약
    st.subheader(f"📊 Statistics Summary ({selected_layer})")
    summary_list = []
    for src in display_df['Source'].unique():
        sub = display_df[display_df['Source'] == src]
        summary_list.append({
            "File": src, "Avg": sub['Value'].mean(), "3-Sigma": sub['Value'].std()*3,
            "Count": len(sub)
        })
    st.dataframe(pd.DataFrame(summary_list))

    # [이후 시각화 로직은 display_df를 사용하여 기존과 동일하게 진행...]
    # (생략: 기존 코드의 시각화 부분에서 plot_df를 display_df 기반으로 필터링하여 사용)
=======
st.set_page_config(page_title="NLX Professional Analyzer", layout="wide")
st.title("🔬 NLX Bump Analysis Dashboard (IQR Advanced)")

# 사이드바: IQR 옵션 세분화
st.sidebar.header("📁 Data & Filtering")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
scale = st.sidebar.number_input("Global Scale Factor (mm to um = 1000)", value=1000)

st.sidebar.subheader("🛡️ Outlier Removal Settings")
use_val_iqr = st.sidebar.checkbox("Apply IQR to Value (H/R/S)", value=True)
use_pitch_iqr = st.sidebar.checkbox("Apply IQR to Pitch (X/Y)", value=True) # 추가된 옵션

if uploaded_files:
    all_data = []
    summary_list = []

    for file in uploaded_files:
        raw_df = pd.read_csv(file)
        # 피치 IQR 옵션 전달
        p_df, d_type = process_data(raw_df, scale, use_val_iqr, use_pitch_iqr)
        
        if p_df is not None:
            p_df['Source'] = file.name
            all_data.append(p_df)
            
            v = p_df['Value'].dropna()
            xp = p_df['X_Pitch'].dropna()
            yp = p_df['Y_Pitch'].dropna()
            
            summary_list.append({
                "File": file.name, "Type": d_type, 
                "Avg": v.mean(), "3-Sigma": v.std()*3,
                "X_Pitch Avg": xp.mean(), "X_Pitch 3σ": xp.std()*3,
                "Y_Pitch Avg": yp.mean(), "Y_Pitch 3σ": yp.std()*3,
                "Count": len(v)
            })

    combined_df = pd.concat(all_data)
    
    # 상단 요약 요약 (Pitch IQR 반영됨)
    st.subheader("📊 Statistics Summary (IQR Applied)")
    st.dataframe(pd.DataFrame(summary_list).style.highlight_min(axis=0, subset=['3-Sigma', 'X_Pitch 3σ']))

    # 상세 분석 대상 선택
    target_file = st.selectbox("Select File for Detail View", [f.name for f in uploaded_files])
    plot_df = combined_df[combined_df['Source'] == target_file]

    # 상세 수치 테이블
    st.markdown("---")
    st.write(f"### 🔢 Detailed Numerical Report: {target_file}")
    col_stat1, col_stat2 = st.columns([1, 2])
    with col_stat1:
        st.write("**Pitch Statistics (Filtered)**")
        p_stats = plot_df[['X_Pitch', 'Y_Pitch']].describe().loc[['mean', 'std']]
        p_stats.loc['3-Sigma'] = p_stats.loc['std'] * 3
        st.table(p_stats)

    # --- [3] 시각화 커스터마이징 및 실행 ---
    st.subheader("🎨 Plot Settings")
    c1, c2, c3, c4 = st.columns(4)
    plots_meta = {
        "Contour": {"title": f"{d_type} Map", "xl": "X (um)", "yl": "Y (um)"},
        "Histogram": {"title": f"{d_type} Dist", "xl": "Value", "yl": "Freq"},
        "Pitch": {"title": "Pitch Spread (IQR Applied)", "xl": "Axis", "yl": "Pitch (um)"},
        "Boxplot": {"title": "Total Comparison", "xl": "File", "yl": "Value"}
    }
    
    config = {}
    for i, (k, v) in enumerate(plots_meta.items()):
        with [c1, c2, c3, c4][i]:
            t = st.text_input(f"Title ({k})", v['title'])
            xl = st.text_input(f"X ({k})", v['xl'])
            yl = st.text_input(f"Y ({k})", v['yl'])
            m_sc = st.checkbox(f"Manual Scale ({k})")
            y_lim = None
            if m_sc:
                # 피치 그래프일 경우 피치 데이터 기준으로 초기값 설정
                ref_data = plot_df['X_Pitch'] if k == "Pitch" else plot_df['Value']
                y_min = st.number_input(f"Min_{k}", value=0.0)
                y_max = st.number_input(f"Max_{k}", value=float(ref_data.max()))
                y_lim = (y_min, y_max)
            config[k] = {"t": t, "xl": xl, "yl": yl, "ylim": y_lim}

    # 그래프 그리기
    st.markdown("---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Contour
    ax1 = axes[0, 0]
    xi = np.linspace(plot_df['X'].min(), plot_df['X'].max(), 100)
    yi = np.linspace(plot_df['Y'].min(), plot_df['Y'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((plot_df['X'], plot_df['Y']), plot_df['Value'], (xi, yi), method='linear')
    cp = ax1.contourf(xi, yi, zi, cmap='viridis', levels=15)
    plt.colorbar(cp, ax=ax1)
    ax1.set_title(config["Contour"]["t"]); ax1.set_xlabel(config["Contour"]["xl"]); ax1.set_ylabel(config["Contour"]["yl"])
    if config["Contour"]["ylim"]: ax1.set_ylim(config["Contour"]["ylim"])

    # 2. Histogram
    ax2 = axes[0, 1]
    sns.histplot(plot_df['Value'], kde=True, ax=ax2, color='skyblue')
    ax2.set_title(config["Histogram"]["t"]); ax2.set_xlabel(config["Histogram"]["xl"]); ax2.set_ylabel(config["Histogram"]["yl"])
    if config["Histogram"]["ylim"]: ax2.set_xlim(config["Histogram"]["ylim"])

    # 3. Pitch Boxplot (IQR 반영된 데이터 사용)
    ax3 = axes[1, 0]
    pitch_melt = plot_df[['X_Pitch', 'Y_Pitch']].melt(var_name='Type', value_name='Pitch')
    sns.boxplot(x='Type', y='Pitch', data=pitch_melt, ax=ax3, palette='Set2')
    ax3.set_title(config["Pitch"]["t"]); ax3.set_xlabel(config["Pitch"]["xl"]); ax3.set_ylabel(config["Pitch"]["yl"])
    if config["Pitch"]["ylim"]: ax3.set_ylim(config["Pitch"]["ylim"])

    # 4. Global Boxplot
    ax4 = axes[1, 1]
    sns.boxplot(x='Source', y='Value', data=combined_df, ax=ax4)
    ax4.set_title(config["Boxplot"]["t"]); ax4.set_xlabel(config["Boxplot"]["xl"]); ax4.set_ylabel(config["Boxplot"]["yl"])
    if config["Boxplot"]["ylim"]: ax4.set_ylim(config["Boxplot"]["ylim"])

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("💡 사이드바에서 CSV 파일들을 업로드하세요.")
>>>>>>> 61426e3a005022eb34196b8b6d3d7fd3319dd467
