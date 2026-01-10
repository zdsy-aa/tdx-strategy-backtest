import os
import pandas as pd
import glob

def update_stock_names():
    # 1. 读取股票列表
    stock_list_path = '/home/ubuntu/upload/中国全部A股股票列表.csv'
    try:
        # 尝试不同的编码读取
        try:
            stock_df = pd.read_csv(stock_list_path, header=None, names=['code', 'name'], dtype={'code': str}, encoding='utf-8')
        except:
            stock_df = pd.read_csv(stock_list_path, header=None, names=['code', 'name'], dtype={'code': str}, encoding='gbk')
        
        # 创建代码到名称的映射
        name_map = dict(zip(stock_df['code'], stock_df['name']))
        print(f"成功读取股票列表，共 {len(name_map)} 条记录")
    except Exception as e:
        print(f"读取股票列表失败: {e}")
        return

    # 2. 遍历 data/day 目录
    base_dir = '/home/ubuntu/tdx-strategy-backtest/data/day'
    markets = ['sh', 'sz', 'bj']
    
    total_updated = 0
    total_files = 0
    
    for market in markets:
        market_dir = os.path.join(base_dir, market)
        if not os.path.exists(market_dir):
            continue
            
        csv_files = glob.glob(os.path.join(market_dir, '*.csv'))
        total_files += len(csv_files)
        
        for file_path in csv_files:
            code = os.path.basename(file_path).replace('.csv', '')
            if code in name_map:
                name = name_map[code]
                try:
                    # 读取原始数据
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    
                    # 如果没有名称列，或者名称不正确，则更新
                    if '名称' not in df.columns or df['名称'].iloc[0] != name:
                        # 在第一列插入名称
                        if '名称' in df.columns:
                            df['名称'] = name
                        else:
                            df.insert(0, '名称', name)
                        
                        # 保存回文件
                        df.to_csv(file_path, index=False, encoding='utf-8-sig')
                        total_updated += 1
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")

    print(f"处理完成。总文件数: {total_files}, 更新文件数: {total_updated}")

if __name__ == "__main__":
    update_stock_names()
