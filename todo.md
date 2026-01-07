# Project TODO

## 基础设施
- [x] 创建 /data/day/ 目录存放日线数据
- [x] 创建 /web/ 目录同步网站代码
- [x] 创建 /py_file/ 目录存放Python脚本
- [x] 创建 /report/ 目录存放报告（总/年/月）
- [x] 推送数据库 schema

## Python 脚本标准化
- [x] indicators.py - 核心指标计算模块
- [x] data_fetcher.py - 数据下载模块
- [x] 101_six_veins_test.py - 六脉神剑单指标测试
- [x] 102_buy_sell_points_test.py - 买卖点单指标测试
- [ ] 103_money_tree_test.py - 摇钱树单指标测试
- [ ] 104_chan_lun_test.py - 缠论买点单指标测试
- [x] 201_steady_combo_test.py - 稳健组合测试
- [x] 202_aggressive_combo_test.py - 激进组合测试
- [ ] 203_resonance_combo_test.py - 共振组合测试
- [ ] 204_custom_combo_test.py - 自定义组合测试框架
- [x] scheduled_data_update.py - 定时更新脚本

## 后端 API 开发
- [ ] 回测报告读取 API
- [ ] AI 策略优化 API
- [ ] 自定义策略回测 API
- [ ] 日线数据查询 API

## 前端功能增强
- [x] 恢复 Home.tsx 原有内容
- [ ] AI 策略优化模块
- [ ] 自定义策略构建器
- [ ] 指标详情页 K 线案例展示
- [ ] 回测报告查看器
- [ ] 报告页面（总/年/月分类）

## 文档与 Git
- [ ] 根目录 README.md
- [ ] /data/ README.md
- [ ] /web/ README.md
- [ ] /py_file/ README.md
- [ ] /report/ README.md
- [ ] Git 同步并确保其他账号可用

## 定时任务
- [x] 每天 15:30 下载当天日线数据脚本
- [ ] 配置系统定时任务
