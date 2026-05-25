from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            import yfinance as yf

            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            tickers_str = params.get('tickers', ['AAPL,TSLA,META,NVDA'])[0]
            tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
            period  = params.get('period', ['3y'])[0]

            # 주가 다운로드
            raw = yf.download(
                tickers, period=period,
                auto_adjust=True, progress=False, threads=False
            )

            # Close 가격 추출 (단일/복수 티커 모두 처리)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw['Close'].copy()
            else:
                prices = raw[['Close']].copy()
                prices.columns = [tickers[0]]

            prices = prices.dropna()
            found  = [t for t in tickers if t in prices.columns]

            if len(found) < 2:
                raise ValueError(f"유효한 데이터가 있는 종목이 2개 미만입니다: {tickers}")

            prices = prices[found]

            if len(prices) < 30:
                raise ValueError("데이터 기간이 너무 짧습니다 (30거래일 미만)")

            daily = prices.pct_change().dropna()

            ann_ret  = {t: float(daily[t].mean() * 252)        for t in found}
            ann_cov  = {r: {c: float(v) for c, v in row.items()}
                        for r, row in (daily.cov() * 252).to_dict().items()}
            corr_mat = {r: {c: float(v) for c, v in row.items()}
                        for r, row in daily.corr().to_dict().items()}
            ann_vol  = {t: float(daily[t].std() * np.sqrt(252)) for t in found}

            body = json.dumps({
                'ok': True,
                'tickers':      found,
                'annual_ret':   ann_ret,
                'annual_cov':   ann_cov,
                'corr':         corr_mat,
                'annual_vol':   ann_vol,
                'trading_days': int(len(daily)),
            }).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type',  'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        except Exception as e:
            err = json.dumps({'ok': False, 'error': str(e)}).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, *args):
        pass  # Vercel 로그 불필요
