"""
双均线策略回测 (Dual Moving Average Strategy Backtest).

策略逻辑:
- 金叉买入: 短期均线上穿长期均线
- 死叉卖出: 短期均线下穿长期均线
"""

from typing import Any

import akquant as aq
import akshare as ak
import numpy as np
from akquant import Bar, Strategy


class DualMAStrategy(Strategy):
    """
    双均线策略.

    参数:
        short_window: 短期均线周期 (默认: 5)
        long_window: 长期均线周期 (默认: 20)
    """

    def __init__(
        self, short_window: int = 5, long_window: int = 20, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.warmup_period = long_window

        self.prev_short_ma: dict[str, float] = {}
        self.prev_long_ma: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol

        closes = self.get_history(count=self.long_window + 1, symbol=symbol, field="close")

        if len(closes) < self.long_window + 1:
            return

        short_ma = np.mean(closes[-self.short_window :])
        long_ma = np.mean(closes[-self.long_window :])

        prev_short = self.prev_short_ma.get(symbol)
        prev_long = self.prev_long_ma.get(symbol)

        current_pos = self.get_position(symbol)

        if prev_short is not None and prev_long is not None:
            if prev_short <= prev_long and short_ma > long_ma:
                if current_pos == 0:
                    self.order_target_percent(symbol=symbol, target_percent=0.95)
                    print(
                        f"[{bar.timestamp_str}] 金叉买入 {symbol}: "
                        f"MA{self.short_window}={short_ma:.2f} > MA{self.long_window}={long_ma:.2f}, "
                        f"Price={bar.close:.2f}"
                    )

            elif prev_short >= prev_long and short_ma < long_ma:
                if current_pos > 0:
                    self.close_position(symbol=symbol)
                    print(
                        f"[{bar.timestamp_str}] 死叉卖出 {symbol}: "
                        f"MA{self.short_window}={short_ma:.2f} < MA{self.long_window}={long_ma:.2f}, "
                        f"Price={bar.close:.2f}"
                    )

        self.prev_short_ma[symbol] = short_ma
        self.prev_long_ma[symbol] = long_ma


if __name__ == "__main__":
    symbol = "sz000001"
    print(f"正在获取 {symbol} 历史数据...")

    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date="20200101", end_date="20241231", adjust="qfq"
    )

    print(f"数据获取完成，共 {len(df)} 条记录")
    print(df.head())

    print("\n开始回测...")
    result = aq.run_backtest(
        data=df,
        strategy=DualMAStrategy,
        symbols=symbol,
        initial_cash=100_000.0,
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        lot_size=100,
    )

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(result)

    print("\n" + "=" * 60)
    print("关键指标")
    print("=" * 60)
    print(f"总收益率: {result.metrics.total_return_pct / 100:.2%}")
    print(f"年化收益率: {result.metrics.annualized_return:.2%}")
    print(f"最大回撤: {-result.metrics.max_drawdown_pct / 100:.2%}")
    print(f"夏普比率: {result.metrics.sharpe_ratio:.2f}")
    print(f"胜率: {result.metrics.win_rate:.2%}")
    print(f"盈亏比: {result.metrics.profit_factor if hasattr(result.metrics, 'profit_factor') else 'N/A'}")

    print("\n生成可视化图表...")
    try:
        result.plot()
    except Exception as e:
        print(f"图表生成失败: {e}")
