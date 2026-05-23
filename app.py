# =============================================================
# 포트폴리오 최적화 웹앱 (Modern Portfolio Theory)
# 실행 방법: streamlit run app.py
# =============================================================

import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import date, timedelta

matplotlib.rcParams['axes.unicode_minus'] = False

# =============================================================
# 페이지 기본 설정
# =============================================================
st.set_page_config(
    page_title="포트폴리오 최적화 분석기",
    page_icon="📈",
    layout="wide",
)

# 전체 배경·기본 글씨색만 CSS로 지정 (클래스 없이 태그 수준)
st.markdown("""
<style>
    .stApp { background-color: #ffffff !important; }
    .stApp, .stApp * { color: #111111; }
    section[data-testid="stSidebar"] { background-color: #f5f7fa !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# 핵심 함수 1: 주가 데이터 수집
# =============================================================
def load_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = {}
    for ticker in tickers:
        df = fdr.DataReader(ticker, start, end)
        if df.empty:
            raise ValueError(f"'{ticker}' 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
        frames[ticker] = df['Close']

    prices = pd.DataFrame(frames).dropna()
    if prices.empty:
        raise ValueError("선택한 기간에 유효한 데이터가 없습니다. 날짜 범위를 넓혀 보세요.")
    return prices


# =============================================================
# 핵심 함수 2: 수익률 및 공분산 계산
# =============================================================
def calc_statistics(prices: pd.DataFrame):
    daily_ret = prices.pct_change().dropna()
    annual_ret = daily_ret.mean() * 252
    annual_cov = daily_ret.cov() * 252
    return daily_ret, annual_ret, annual_cov


# =============================================================
# 핵심 함수 3: 몬테카를로 시뮬레이션
# =============================================================
def run_monte_carlo(
    tickers: list[str],
    annual_ret: pd.Series,
    annual_cov: pd.DataFrame,
    n_sim: int,
    rf: float = 0.03,
) -> pd.DataFrame:
    ret_list, risk_list, weights_list, sharpe_list = [], [], [], []
    for _ in range(n_sim):
        w = np.random.random(len(tickers))
        w = w / w.sum()
        port_return = np.dot(w, annual_ret)
        port_risk = np.sqrt(np.dot(w.T, np.dot(annual_cov, w)))
        sharpe = (port_return - rf) / port_risk
        ret_list.append(port_return)
        risk_list.append(port_risk)
        weights_list.append(w)
        sharpe_list.append(sharpe)

    result = pd.DataFrame(weights_list, columns=tickers)
    result['Return'] = ret_list
    result['Risk'] = risk_list
    result['Sharpe'] = sharpe_list
    return result


# =============================================================
# 핵심 함수 4: 최적 포트폴리오 탐색
# =============================================================
def find_optimal_portfolios(result: pd.DataFrame):
    max_sharpe = result.loc[result['Sharpe'].idxmax()]
    min_risk   = result.loc[result['Risk'].idxmin()]
    return max_sharpe, min_risk


# =============================================================
# 핵심 함수 5: 효율적 투자선 시각화 (라이트 테마)
# =============================================================
def plot_efficient_frontier(result, max_sharpe, min_risk):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fafb')

    sc = ax.scatter(
        result['Risk'], result['Return'],
        c=result['Sharpe'], cmap='RdYlGn', alpha=0.5, s=8,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Sharpe Ratio', color='#333333', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='#333333')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#333333')

    ax.scatter(max_sharpe['Risk'], max_sharpe['Return'],
               c='#e53935', marker='*', s=700, zorder=5,
               label=f"최대 샤프지수  (Sharpe={max_sharpe['Sharpe']:.2f})",
               edgecolors='#111111', linewidths=0.8)

    ax.scatter(min_risk['Risk'], min_risk['Return'],
               c='#1565c0', marker='X', s=400, zorder=5,
               label=f"최소 위험  (Risk={min_risk['Risk']:.2%})",
               edgecolors='#111111', linewidths=0.8)

    ax.set_title('효율적 투자선 (Efficient Frontier)',
                 color='#111111', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('Risk (표준편차)', color='#333333', fontsize=11)
    ax.set_ylabel('연간 기대 수익률', color='#333333', fontsize=11)
    ax.tick_params(colors='#333333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.legend(facecolor='#ffffff', edgecolor='#cccccc',
              labelcolor='#111111', fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.tight_layout()
    return fig


# =============================================================
# 헬퍼: 비중 막대 시각화 (인라인 스타일만 사용)
# =============================================================
def render_weight_blocks(portfolio: pd.Series, tickers: list[str], color: str = "red"):
    if color == "red":
        bar_gradient = "linear-gradient(90deg, #e53935, #ef9a9a)"
        header_gradient = "linear-gradient(90deg, #c62828, #e57373)"
    else:
        bar_gradient = "linear-gradient(90deg, #1565c0, #64b5f6)"
        header_gradient = "linear-gradient(90deg, #0d47a1, #42a5f5)"

    icon = "⭐" if color == "red" else "🛡️"
    title = "최대 샤프지수 포트폴리오" if color == "red" else "최소 위험 포트폴리오"
    subtitle = "위험 대비 수익이 가장 효율적인 공격형" if color == "red" else "변동성이 가장 낮은 안정 추구형"

    # 헤더
    st.markdown(f"""
<div style="background:{header_gradient};border-radius:12px;padding:1rem 1.4rem;
            margin-bottom:1rem;color:#ffffff;">
  <div style="font-size:1.3rem;font-weight:800;">{icon} {title}</div>
  <div style="font-size:0.85rem;font-weight:400;margin-top:0.2rem;">{subtitle}</div>
</div>
""", unsafe_allow_html=True)

    # 종목별 비중 막대
    rows_html = ""
    for t in tickers:
        pct = portfolio[t]
        bar_w = max(pct * 100, 1)
        rows_html += f"""
<div style="display:flex;align-items:center;background:#ffffff;border:1px solid #e0e0e0;
            border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.45rem;">
  <span style="font-size:1rem;font-weight:700;color:#1a237e;
               min-width:72px;">{t}</span>
  <div style="flex:1;background:#e8eaf6;border-radius:6px;height:16px;
              margin:0 0.8rem;overflow:hidden;">
    <div style="width:{bar_w:.1f}%;height:100%;background:{bar_gradient};
                border-radius:6px;"></div>
  </div>
  <span style="font-size:1.15rem;font-weight:800;color:#111111;
               min-width:56px;text-align:right;">{pct:.1%}</span>
</div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    # 요약 지표 (st.columns + st.metric 사용 → HTML 오류 없음)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("기대 수익률", f"{portfolio['Return']:.2%}")
    with c2:
        st.metric("리스크 (표준편차)", f"{portfolio['Risk']:.2%}")
    with c3:
        st.metric("샤프지수", f"{portfolio['Sharpe']:.4f}")


# =============================================================
# 메인 UI
# =============================================================
def main():

    # ── 타이틀 ──────────────────────────────────────────────
    st.markdown(
        '<h1 style="text-align:center;color:#1a237e;font-size:2.5rem;font-weight:800;">'
        '📈 포트폴리오 최적화 분석기</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:#455a64;font-size:1.05rem;margin-bottom:1.5rem;">'
        '현대 포트폴리오 이론(MPT) 기반 · 몬테카를로 시뮬레이션 · 효율적 투자선 시각화</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 사이드바 ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        st.markdown("---")

        st.subheader("1️⃣ 종목 입력")
        ticker_input = st.text_area(
            "티커 심볼을 쉼표로 구분해서 입력하세요",
            value="AAPL, TSLA, META, NVDA",
            height=80,
            help="미국 주식은 영문 티커, 한국 주식은 종목코드(예: 005930)를 입력하세요.",
        )
        st.markdown("---")

        st.subheader("2️⃣ 분석 기간")
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input(
                "시작일",
                value=date.today() - timedelta(days=365 * 3),
                max_value=date.today() - timedelta(days=30),
            )
        with col_e:
            end_date = st.date_input(
                "종료일",
                value=date.today(),
                max_value=date.today(),
            )
        st.markdown("---")

        st.subheader("3️⃣ 무위험 수익률")
        rf_rate = st.slider(
            "Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.1,
            help="미국 국채 10년물 수익률 기준. 샤프지수 계산에 사용됩니다.",
        ) / 100
        st.markdown("---")

        st.subheader("4️⃣ 시뮬레이션 횟수")
        n_sim = st.select_slider(
            "횟수가 많을수록 정확하지만 느려집니다",
            options=[10_000, 20_000, 30_000, 40_000, 50_000],
            value=20_000,
        )
        st.markdown("---")

        run_btn = st.button("🚀 분석 시작", use_container_width=True, type="primary")

    # ── 실행 전 안내 ───────────────────────────────────────────
    if not run_btn:
        st.info(
            "👈 왼쪽 사이드바에서 **종목 · 기간 · 시뮬레이션 횟수**를 설정한 뒤 "
            "**'🚀 분석 시작'** 버튼을 눌러주세요.",
            icon="💡",
        )
        with st.expander("📖 사용 방법 & 용어 설명 보기"):
            st.markdown("""
**사용 방법**
1. 왼쪽 사이드바에 분석할 종목의 티커 심볼을 입력합니다.
2. 분석 기간(시작일 ~ 종료일)을 선택합니다.
3. 무위험 수익률(보통 국채 금리)을 설정합니다.
4. 몬테카를로 시뮬레이션 횟수를 선택합니다.
5. **'🚀 분석 시작'** 버튼을 클릭하면 결과가 출력됩니다.

---

| 용어 | 설명 |
|------|------|
| **기대 수익률** | 포트폴리오가 1년 동안 벌어들일 것으로 예상되는 수익률 |
| **리스크** | 수익률의 표준편차. 값이 클수록 변동성이 큽니다 |
| **샤프지수** | (수익률 - 무위험수익률) ÷ 리스크. 위험 대비 수익 효율성 |
| **최대 샤프 포트폴리오** | 위험 1단위당 수익이 가장 높은 포트폴리오 (★ 빨간 별) |
| **최소 위험 포트폴리오** | 변동성이 가장 낮은 보수적인 포트폴리오 (✕ 파란 X) |
| **효율적 투자선** | 동일 위험 대비 최대 수익을 내는 포트폴리오들의 집합 |
            """)
        return

    # ── 분석 실행 ─────────────────────────────────────────────
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("종목을 최소 2개 이상 입력해 주세요.")
        return
    if start_date >= end_date:
        st.error("시작일이 종료일보다 이전이어야 합니다.")
        return

    progress_bar = st.progress(0, text="데이터를 불러오는 중...")
    try:
        progress_bar.progress(15, text=f"📡 {', '.join(tickers)} 주가 데이터 수집 중...")
        prices = load_stock_data(tickers, str(start_date), str(end_date))

        progress_bar.progress(35, text="📊 수익률 및 공분산 계산 중...")
        _, annual_ret, annual_cov = calc_statistics(prices)

        progress_bar.progress(50, text=f"🎲 {n_sim:,}회 몬테카를로 시뮬레이션 실행 중...")
        result = run_monte_carlo(tickers, annual_ret, annual_cov, n_sim, rf=rf_rate)

        progress_bar.progress(85, text="🔍 최적 포트폴리오 탐색 중...")
        max_sharpe, min_risk = find_optimal_portfolios(result)

        progress_bar.progress(100, text="✅ 분석 완료!")
        progress_bar.empty()

    except ValueError as e:
        progress_bar.empty()
        st.error(f"❌ 오류: {e}")
        return
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        return

    st.success(
        f"✅ 분석 완료! | 종목: {', '.join(tickers)} | "
        f"기간: {start_date} ~ {end_date} | "
        f"데이터: {len(prices)}거래일 | 시뮬레이션: {n_sim:,}회"
    )
    st.divider()

    # ================================================================
    # ★★★ 핵심 결과: 두 최적 포트폴리오 투자 비중 ★★★
    # ================================================================
    st.markdown(
        '<h2 style="color:#111111;font-size:1.7rem;font-weight:800;margin-bottom:0.3rem;">'
        '💼 최적 포트폴리오 투자 비중 안내</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#333333;font-size:1rem;margin-bottom:1.2rem;">'
        '아래 두 포트폴리오 중 <b>투자 성향에 맞는 비중으로 투자</b>하세요. '
        '비중은 전체 투자금액 대비 각 종목에 배분할 비율입니다.</p>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        render_weight_blocks(max_sharpe, tickers, color="red")
        st.markdown(
            '<p style="font-weight:700;color:#111111;margin-top:0.8rem;">종목별 정확한 투자 비중</p>',
            unsafe_allow_html=True,
        )
        weight_table_ms = pd.DataFrame([
            {
                "종목": t,
                "투자 비중": f"{max_sharpe[t]:.2%}",
                "1,000만원 기준 (원)": f"{int(max_sharpe[t] * 10_000_000):,}",
            }
            for t in tickers
        ])
        st.dataframe(weight_table_ms, use_container_width=True, hide_index=True)

    with right_col:
        render_weight_blocks(min_risk, tickers, color="blue")
        st.markdown(
            '<p style="font-weight:700;color:#111111;margin-top:0.8rem;">종목별 정확한 투자 비중</p>',
            unsafe_allow_html=True,
        )
        weight_table_mr = pd.DataFrame([
            {
                "종목": t,
                "투자 비중": f"{min_risk[t]:.2%}",
                "1,000만원 기준 (원)": f"{int(min_risk[t] * 10_000_000):,}",
            }
            for t in tickers
        ])
        st.dataframe(weight_table_mr, use_container_width=True, hide_index=True)

    st.divider()

    # ── 효율적 투자선 ──────────────────────────────────────────
    st.markdown("### 📉 효율적 투자선 (Efficient Frontier)")
    st.caption("붉은 별(★) = 최대 샤프지수 포트폴리오 / 파란 X = 최소 위험 포트폴리오")
    fig = plot_efficient_frontier(result, max_sharpe, min_risk)
    st.pyplot(fig, use_container_width=True)

    st.divider()

    # ── 종목별 연간 기대 수익률 비교 ─────────────────────────
    st.markdown("### 📊 종목별 연간 기대 수익률 비교")
    ret_compare = pd.DataFrame({
        "종목": tickers,
        "연간 기대 수익률": [annual_ret[t] for t in tickers],
    }).set_index("종목")
    st.bar_chart(ret_compare, height=280)

    # ── 상관관계 히트맵 ───────────────────────────────────────
    st.markdown("### 🔗 종목 간 상관관계")
    st.caption("값이 1에 가까울수록 두 종목이 함께 움직이며, 분산 효과가 낮습니다.")

    corr = prices.pct_change().dropna().corr()
    fig2, ax2 = plt.subplots(
        figsize=(max(4, len(tickers) * 1.2), max(3, len(tickers)))
    )
    fig2.patch.set_facecolor('#ffffff')
    ax2.set_facecolor('#f9fafb')

    im = ax2.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(tickers)))
    ax2.set_yticks(range(len(tickers)))
    ax2.set_xticklabels(tickers, color='#111111')
    ax2.set_yticklabels(tickers, color='#111111')
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax2.text(j, i, f"{corr.values[i, j]:.2f}",
                     ha='center', va='center',
                     color='white' if abs(corr.values[i, j]) > 0.6 else '#111111',
                     fontsize=11, fontweight='bold')
    fig2.colorbar(im, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.divider()

    # ── 데이터 내보내기 ───────────────────────────────────────
    st.markdown("### 💾 데이터 내보내기")
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
        st.download_button(
            label="📥 시뮬레이션 전체 결과 (CSV)",
            data=result.to_csv(index=False).encode('utf-8-sig'),
            file_name="simulation_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        summary = pd.DataFrame([max_sharpe, min_risk])
        summary.insert(0, '구분', ['최대 샤프지수', '최소 위험'])
        st.download_button(
            label="📥 최적 포트폴리오 요약 (CSV)",
            data=summary.to_csv(index=False).encode('utf-8-sig'),
            file_name="optimal_portfolios.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        '<div style="text-align:center;color:#90a4ae;font-size:0.82rem;margin-top:2rem;">'
        '현대 포트폴리오 이론(MPT) 기반 분석기 &nbsp;|&nbsp; '
        '데이터 출처: FinanceDataReader &nbsp;|&nbsp; '
        '본 결과는 투자 권유가 아닙니다.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
