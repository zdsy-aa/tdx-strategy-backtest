import os
import csv
import re

# 定义输入和输出目录
input_dir = './'  # 当前目录中的.txt文件
output_dir = './data/day/'  # CSV文件保存路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
for market in ['bj', 'sz', 'sh']:
    os.makedirs(os.path.join(output_dir, market), exist_ok=True)

# 定义表头，按照目标顺序排列
headers = ['名称', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

# 根据股票代码判断市场
def get_market(stock_code):
    """根据股票代码判断市场"""
    if stock_code.startswith(("60", "68", "90", "11", "5")):
        return "sh"  # 上海市场
    elif stock_code.startswith(("00", "30", "20", "12", "15")):
        return "sz"  # 深圳市场
    elif stock_code.startswith(("8", "4", "92")):
        return "bj"  # 北京市场
    return "sh"  # 默认上海市场

# 提取股票名称
def extract_stock_name(line):
    """提取股票名称
    第一行格式示例: 000006 深振业Ａ 日线 前复权
    """
    parts = line.strip().split()
    if len(parts) >= 2:
        return parts[1]
    return ""

# 转换每个txt文件的函数
def convert_txt_to_csv(txt_file):
    filename = os.path.basename(txt_file)
    stock_code = os.path.splitext(filename)[0]  # 使用文件名作为股票代码
    market = get_market(stock_code)  # 根据股票代码获取市场
    csv_file_path = os.path.join(output_dir, market, f'{stock_code}.csv')

    # 尝试使用GBK编码打开文件
    try:
        with open(txt_file, 'r', encoding='gbk') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"读取文件失败 {txt_file}: {e}")
        return

    if not lines:
        print(f"文件为空: {txt_file}")
        return

    # 第一行是说明行，提取股票名称
    stock_name = extract_stock_name(lines[0])

    # 准备数据
    rows = []
    # 从第三行开始是数据 (第一行说明，第二行表头)
    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        columns = line.split(';')
        # 确保每行数据至少包含7个有效的列（日期、开盘、最高、最低、收盘、成交量、成交额）
        if len(columns) >= 7:
            try:
                # 将成交量除以 100
                volume = float(columns[5]) / 100
                
                # 按照目标顺序重排数据
                row = [
                    stock_name,  # 股票名称
                    columns[0],  # 日期
                    columns[1],  # 开盘
                    columns[4],  # 收盘
                    columns[2],  # 最高
                    columns[3],  # 最低
                    volume,      # 成交量
                    columns[6],  # 成交额
                    '',  # 振幅
                    '',  # 涨跌幅
                    '',  # 涨跌额
                    '',  # 换手率
                ]
                rows.append(row)
            except ValueError:
                continue

    if rows:
        # 将数据写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='gbk') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)  # 写入表头
            writer.writerows(rows)  # 写入数据行
        print(f"已将 {filename} 转换为 {csv_file_path}")
    else:
        print(f"文件 {filename} 没有有效数据，未生成CSV文件。")

# 遍历当前目录下的所有.txt文件并转换
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        txt_file_path = os.path.join(input_dir, filename)
        convert_txt_to_csv(txt_file_path)

print("所有.txt文件转换完成。")
