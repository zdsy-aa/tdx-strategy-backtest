import json
import os
from pathlib import Path

# 路径定义
BASE_DIR = Path(__file__).parent.parent
BACKTEST_RESULTS_FILE = BASE_DIR / "data" / "backtest_results" / "backtest_results.json"
STRATEGIES_JSON_FILE = BASE_DIR / "web" / "client" / "src" / "data" / "strategies.json"

def update_strategies():
    if not BACKTEST_RESULTS_FILE.exists():
        print(f"错误: 找不到回测结果文件 {BACKTEST_RESULTS_FILE}")
        return

    if not STRATEGIES_JSON_FILE.exists():
        print(f"错误: 找不到前端数据文件 {STRATEGIES_JSON_FILE}")
        return

    # 读取回测结果
    with open(BACKTEST_RESULTS_FILE, 'r', encoding='utf-8') as f:
        backtest_data = json.load(f)

    # 读取前端 strategies.json
    with open(STRATEGIES_JSON_FILE, 'r', encoding='utf-8') as f:
        web_data = json.load(f)

    # 1. 更新单指标策略 (singleIndicatorStrategies)
    for strategy in web_data.get('singleIndicatorStrategies', []):
        s_id = strategy['id']
        # 映射 ID
        mapped_id = s_id
        if s_id == 'chan_lun_2buy': mapped_id = 'chan_buy2' # 缠论二买映射
        if s_id == 'money_tree_buy': mapped_id = 'money_tree' # 摇钱树映射
        
        if mapped_id in backtest_data:
            res = backtest_data[mapped_id]
            stats = res['stats']['total']
            strategy['stats']['total'] = {
                "winRate": f"{stats['win_rate']}%",
                "avgReturn": f"{stats['avg_return']}%",
                "optimalPeriod": f"{res['optimal_period_win']}天",
                "trades": stats['trades']
            }
            # 更新年度和月度数据
            strategy['stats']['yearly'] = res['stats']['yearly']
            strategy['stats']['monthly'] = res['stats']['monthly']
            print(f"已更新单指标策略: {s_id}")

    # 2. 更新组合方案 (strategies)
    # 注意：组合方案的数据目前在 full_backtest.py 中也有计算，但 ID 可能不同
    # 这里我们尝试匹配
    for strategy in web_data.get('strategies', []):
        s_id = strategy['id']
        # 映射组合 ID
        mapped_id = None
        if s_id == 'steady': mapped_id = 'combo_steady'
        if s_id == 'aggressive': mapped_id = 'combo_aggressive'
        if s_id == 'resonance': mapped_id = 'combo_resonance'
        
        if mapped_id and mapped_id in backtest_data:
            res = backtest_data[mapped_id]
            stats = res['stats']['total']
            strategy['stats']['total'] = {
                "winRate": f"{stats['win_rate']}%",
                "avgReturn": f"{stats['avg_return']}%",
                "optimalPeriod": f"{res['optimal_period_win']}天",
                "trades": stats['trades']
            }
            strategy['stats']['yearly'] = res['stats']['yearly']
            strategy['stats']['monthly'] = res['stats']['monthly']
            print(f"已更新组合方案: {s_id}")

    # 保存更新后的文件
    with open(STRATEGIES_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(web_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功更新 {STRATEGIES_JSON_FILE}")

if __name__ == "__main__":
    update_strategies()
