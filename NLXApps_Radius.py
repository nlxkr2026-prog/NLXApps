import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import os

# 1. Radius 데이터 전처리 및 Pitch 계산 함수
def process_radius_data(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    # 데이터 로드 및 결측치 제거
    df = pd.read_csv(file_path).dropna(subset=['Bump_Center_X', 'Bump_Center_Y', 'Radius'])
    df.columns = [c.strip() for c in df.columns] 
    
    # [A] 단위 변환 (mm -> um)
    df['X'] = df['Bump_Center_X'] * 1000
    df['Y'] = df['Bump_Center_Y'] * 1000
    df['Radius'] = df['Radius'] * 1000
    
    # [B] Radius 이상치 제거 (IQR 필터링)
    df_clean = df[df['Radius'] != 0].copy()
    qr1, qr3 = df_clean['Radius'].quantile([0.25, 0.75])
    iqr_r = qr3 - qr1
    df_final = df_clean[
        (df_clean['Radius'] >= qr1 - 1.5 * iqr_r) & 
        (df_clean['Radius'] <= qr3 + 1.5 * iqr_r)
    ].copy()

    # [C] 양산형 Pitch 계산 (좌표 그리드 기반)
    df_final['Y_grid'] = df_final['Y'].round(0) 
    df_final = df_final.sort_values(by=['Y_grid', 'X'])
    df_final['X_Pitch'] = df_final.groupby('Y_grid')['X'].diff()

    df_final['X_grid'] = df_final['X'].round(0)
    df_final = df_final.sort_values(by=['X_grid', 'Y'])
    df_final['Y_Pitch'] = df_final.groupby('X_grid')['Y'].diff()

    # [D] Pitch 이상치 제거 (IQR)
    for col in ['X_Pitch', 'Y_Pitch']:
        valid_data = df_final[col].dropna()
        if not valid_data.empty:
            q1, q3 = valid_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            df_final.loc[(df_final[col] < q1 - 1.5 * iqr) | (df_final[col] > q3 + 1.5 * iqr), col] = np.nan

    return df_final

# 2. 통계치 출력 함수 (3-Sigma 추가)
def print_radius_statistics(df):
    if df is None: return
    
    # 평균, 표준편차 계산 및 3시그마 산출
    items = ["Radius", "X_Pitch", "Y_Pitch"]
    
    print("="*75)
    print(f"{'Item':<12} | {'Average (um)':<15} | {'Std Dev (um)':<15} | {'3-Sigma (um)':<15}")
    print("-" * 75)
    
    for item in items:
        data = df[item].dropna()
        avg = data.mean()
        std_dev = data.std()
        three_sigma = std_dev * 3
        
        print(f"{item:<12} | {avg:>15.6f} | {std_dev:>15.6f} | {three_sigma:>15.6f}")
    print("="*75)

# 3. 시각화 함수
def plot_radius_visualizations(df):
    if df is None: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [1] Radius Contour Map
    ax1 = axes[0, 0]
    xi = np.linspace(df['X'].min(), df['X'].max(), 200)
    yi = np.linspace(df['Y'].min(), df['Y'].max(), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((df['X'], df['Y']), df['Radius'], (xi, yi), method='linear')
    cp = ax1.contourf(xi, yi, zi, cmap='plasma', levels=15)
    fig.colorbar(cp, ax=ax1, label='Radius (um)')
    ax1.set_title('Bump Radius Map (um)')

    # [2] Radius Box Plot
    sns.boxplot(y=df['Radius'], ax=axes[0, 1], color='orchid')
    axes[0, 1].set_title('Radius Distribution (um)')

    # [3] X-Pitch Box Plot
    sns.boxplot(y=df['X_Pitch'].dropna(), ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('X-Pitch Distribution (um)')

    # [4] Y-Pitch Box Plot
    sns.boxplot(y=df['Y_Pitch'].dropna(), ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title('Y-Pitch Distribution (um)')

    plt.tight_layout()
    plt.show()

# --- 메인 실행부 ---
if __name__ == "__main__":
    FOLDER = 'C:/Users/KSJEOKI1/OneDrive - Carl Zeiss AG/문서/Other Demo/Astar'
    FILE = 'single_cross_section_radii_raw_data.csv'

    # 1. 데이터 처리
    radius_df = process_radius_data(FOLDER, FILE)
    
    # 2. 통계 출력 (평균 및 편차)
    if radius_df is not None:
        print_radius_statistics(radius_df)
        
        # 3. 그래프 출력
        plot_radius_visualizations(radius_df)