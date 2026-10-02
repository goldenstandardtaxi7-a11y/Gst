"""Minimal Alpaca REST client (trading API v2 + market data v2)."""
import pandas as pd
import requests

from .config import BrokerConfig


class AlpacaBroker:
    def __init__(self, cfg: BrokerConfig, session=None, timeout=30):
        self.cfg = cfg
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": cfg.api_key,
                                     "APCA-API-SECRET-KEY": cfg.secret_key})

    def _request(self, method, url, **kwargs):
        resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {url} -> {resp.status_code}: {resp.text}")
        return resp.json() if resp.content else None

    def _trading(self, method, path, **kwargs):
        return self._request(method, self.cfg.trading_url + path, **kwargs)

    def account(self):
        return self._trading("GET", "/v2/account")

    def clock(self):
        return self._trading("GET", "/v2/clock")

    def positions(self):
        """{symbol: market_value} for open positions."""
        return {p["symbol"]: float(p["market_value"]) for p in self._trading("GET", "/v2/positions")}

    def submit_notional_order(self, symbol, side, notional):
        return self._trading("POST", "/v2/orders", json={
            "symbol": symbol, "side": side, "type": "market", "time_in_force": "day",
            "notional": f"{notional:.2f}"})

    def close_position(self, symbol):
        return self._trading("DELETE", f"/v2/positions/{symbol}")

    def cancel_all_orders(self):
        return self._trading("DELETE", "/v2/orders")

    def close_all_positions(self):
        return self._trading("DELETE", "/v2/positions", params={"cancel_orders": "true"})

    def daily_closes(self, symbols, start, end=None) -> pd.DataFrame:
        """Split/dividend-adjusted daily closes, one column per symbol."""
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start,
                  "adjustment": "all", "feed": "iex", "limit": 10000}
        if end:
            params["end"] = end
        rows = {}
        while True:
            data = self._request("GET", self.cfg.data_url + "/v2/stocks/bars", params=params)
            for sym, bars in (data.get("bars") or {}).items():
                for bar in bars:
                    rows.setdefault(sym, {})[bar["t"][:10]] = float(bar["c"])
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()[[s for s in symbols if s in df.columns]]
