import { useState, useMemo } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Scatter, Bar } from 'recharts';
import stockReportsData from "@/data/stock_reports.json";

interface StockReport {
  code: string;
  name: string;
  market: string;
  marketName: string;
}

interface KLineData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal?: string;
  signalType?: 'buy' | 'sell';
}

interface TradePair {
  buy: KLineData;
  sell: KLineData;
  profit: number;
  profitPercent: string;
  amount: number;
}

type DateRangeType = 'year' | 'month' | 'custom';

export default function VisualBuyPoints() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStock, setSelectedStock] = useState<string>("");
  const [marketFilter, setMarketFilter] = useState<"all" | "sh" | "sz" | "bj">("all");
  const [signalFilter, setSignalFilter] = useState<"all" | "buy" | "sell">("all");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [dateRangeType, setDateRangeType] = useState<DateRangeType>('month');
  const [selectedYear, setSelectedYear] = useState<string>("");
  const [selectedMonth, setSelectedMonth] = useState<string>("");
  const [customStartDate, setCustomStartDate] = useState("");
  const [customEndDate, setCustomEndDate] = useState("");

  // 获取股票列表
  const stockReports: StockReport[] = useMemo(() => {
    return stockReportsData as StockReport[];
  }, []);

  // 模糊搜索股票列表
  const filteredStocks = useMemo(() => {
    if (!searchTerm) return [];
    
    return stockReports
      .filter(stock => {
        const searchLower = searchTerm.toLowerCase();
        const matchCode = stock.code.includes(searchTerm);
        const matchName = stock.name.toLowerCase().includes(searchLower);
        const matchMarket = marketFilter === "all" || stock.market === marketFilter;
        return (matchCode || matchName) && matchMarket;
      })
      .slice(0, 20);
  }, [stockReports, searchTerm, marketFilter]);

  // 模拟K线数据生成（2023-2026年，共3年数据）
  const generateMockKLineData = (stockCode: string): KLineData[] => {
    const data: KLineData[] = [];
    let basePrice = 10 + Math.random() * 20;
    const startDate = new Date('2023-01-01');
    const endDate = new Date('2026-01-10'); // 到今天
    let lastSignalIndex = -10;
    let lastSignalType: 'buy' | 'sell' | undefined;
    
    let currentDate = new Date(startDate);
    let dayIndex = 0;
    
    while (currentDate <= endDate) {
      // 跳过周末
      if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
        // 生成当天的价格波动
        const change = (Math.random() - 0.5) * 2;
        const open = basePrice;
        const close = basePrice + change;
        const high = Math.max(open, close) + Math.random() * 1;
        const low = Math.min(open, close) - Math.random() * 1;
        const volume = Math.floor(Math.random() * 1000000) + 100000;
        
        let signal: string | undefined;
        let signalType: 'buy' | 'sell' | undefined;
        
        // 确保买卖信号交替出现，且有足够间隔
        if (dayIndex - lastSignalIndex >= 5 && Math.random() > 0.88) {
          if (lastSignalType === 'buy') {
            signal = "卖点１";
            signalType = "sell";
          } else {
            const buySignals = ["六脉６红", "买点２", "缠论一买"];
            signal = buySignals[Math.floor(Math.random() * buySignals.length)];
            signalType = "buy";
          }
          lastSignalIndex = dayIndex;
          lastSignalType = signalType;
        }
        
        data.push({
          date: currentDate.toISOString().split('T')[0],
          open: parseFloat(open.toFixed(2)),
          high: parseFloat(high.toFixed(2)),
          low: parseFloat(low.toFixed(2)),
          close: parseFloat(close.toFixed(2)),
          volume,
          signal,
          signalType
        });
        
        basePrice = close;
        dayIndex++;
      }
      
      // 移动到下一天
      currentDate.setDate(currentDate.getDate() + 1);
    }
    
    return data;
  };

  // 获取选中股票的K线数据
  const klineData = useMemo(() => {
    if (!selectedStock) return [];
    return generateMockKLineData(selectedStock);
  }, [selectedStock]);

  // 从K线数据中提取可用的年份和月份
  const availableYears = useMemo(() => {
    if (klineData.length === 0) return [];
    const years = Array.from(new Set(klineData.map(d => d.date.substring(0, 4)))).sort().reverse();
    return years;
  }, [klineData]);

  const availableMonths = useMemo(() => {
    if (klineData.length === 0 || !selectedYear) return [];
    const months = Array.from(
      new Set(
        klineData
          .filter(d => d.date.startsWith(selectedYear))
          .map(d => d.date.substring(5, 7))
      )
    ).sort().reverse();
    return months;
  }, [klineData, selectedYear]);

  // 不自动初始化年月选择，保持默认为空

  // 根据日期范围筛选数据（默认不筛选，显示全部数据）
  const filteredKLineData = useMemo(() => {
    if (klineData.length === 0) return [];
    
    // 如果没有选择任何日期筛选条件，返回全部数据
    if (!selectedYear && !customStartDate && !customEndDate) {
      return klineData;
    }
    
    switch (dateRangeType) {
      case 'year':
        if (selectedYear) {
          return klineData.filter(d => d.date.startsWith(selectedYear));
        }
        return klineData;
      case 'month':
        if (selectedYear && selectedMonth) {
          const yearMonth = `${selectedYear}-${selectedMonth}`;
          return klineData.filter(d => d.date.startsWith(yearMonth));
        }
        return klineData;
      case 'custom':
        if (customStartDate && customEndDate) {
          return klineData.filter(d => d.date >= customStartDate && d.date <= customEndDate);
        }
        return klineData;
      default:
        return klineData;
    }
  }, [klineData, dateRangeType, selectedYear, selectedMonth, customStartDate, customEndDate]);

  // 买卖匹配逻辑 (FIFO)，修复负交易金额
  const tradePairs = useMemo(() => {
    const pairs: TradePair[] = [];
    const buyQueue: KLineData[] = [];
    
    filteredKLineData.forEach(day => {
      if (day.signalType === 'buy') {
        buyQueue.push(day);
      } else if (day.signalType === 'sell') {
        if (buyQueue.length > 0) {
          const buyDay = buyQueue.shift()!;
          const profit = parseFloat((day.close - buyDay.close).toFixed(2));
          const profitPercent = ((profit / buyDay.close) * 100).toFixed(2);
          // 修复：交易金额应该是买入价格 * 100股（1手），而不是差价
          const amount = parseFloat((buyDay.close * 100).toFixed(2));
          pairs.push({ 
            buy: buyDay, 
            sell: day,
            profit,
            profitPercent,
            amount
          });
        }
      }
    });
    
    console.log(`交易对数量: ${pairs.length}`);
    console.log('交易对详情:', pairs.map(p => `${p.buy.date} -> ${p.sell.date} (${p.profitPercent}%)`));
    
    return pairs;
  }, [filteredKLineData]);

  // 获取选中股票的信息
  const selectedStockInfo = useMemo(() => {
    return stockReports.find(stock => stock.code === selectedStock);
  }, [stockReports, selectedStock]);

  // 处理股票选择
  const handleStockSelect = (code: string) => {
    setSelectedStock(code);
    setSearchTerm('');
    setShowSuggestions(false);
    // 重置日期选择
    setSelectedYear("");
    setSelectedMonth("");
  };

  // 自定义Tooltip
  const CustomTooltip = (props: any) => {
    const { active, payload } = props;
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
          <p className="text-white font-semibold mb-2">{data.date}</p>
          <div className="text-sm space-y-1">
            <p className="text-gray-300">开盘: <span className="text-white font-medium">{data.open}</span></p>
            <p className="text-gray-300">收盘: <span className="text-white font-medium">{data.close}</span></p>
            <p className="text-gray-300">最高: <span className="text-white font-medium">{data.high}</span></p>
            <p className="text-gray-300">最低: <span className="text-white font-medium">{data.low}</span></p>
            <p className="text-gray-300">成交量: <span className="text-white font-medium">{(data.volume / 10000).toFixed(0)}万</span></p>
            {data.signal && (
              <p className={`font-semibold mt-2 pt-2 border-t border-gray-600 ${
                data.signalType === 'buy' ? 'text-red-400' : 'text-green-400'
              }`}>
                📍 {data.signal}
              </p>
            )}
          </div>
        </div>
      );
    }
    return null;
  };



  return (
    <Layout>
      <div className="container mx-auto p-6 space-y-6">
        {/* 页面标题 */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">可视化买点</h1>
            <p className="text-gray-400">K线图展示与信号标注 | 红色⚪买入 | 绿色▲卖出</p>
          </div>
        </div>

        {/* 股票筛选区域 */}
        <Card className="bg-gray-900 border-gray-800">
          <CardHeader>
            <CardTitle className="text-white">股票筛选</CardTitle>
            <CardDescription>输入股票代码或名称进行模糊搜索</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* 搜索框 */}
              <div className="relative md:col-span-2">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 z-10" />
                <Input
                  placeholder="搜索股票代码或名称（如：600000 或 浦发银行）"
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setShowSuggestions(true);
                    setFocusedIndex(-1);
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      if (focusedIndex >= 0 && focusedIndex < filteredStocks.length) {
                        handleStockSelect(filteredStocks[focusedIndex].code);
                      } else if (filteredStocks.length === 1) {
                        handleStockSelect(filteredStocks[0].code);
                      }
                    } else if (e.key === 'ArrowDown') {
                      e.preventDefault();
                      setFocusedIndex(prev => Math.min(prev + 1, filteredStocks.length - 1));
                    } else if (e.key === 'ArrowUp') {
                      e.preventDefault();
                      setFocusedIndex(prev => Math.max(prev - 1, -1));
                    } else if (e.key === 'Escape') {
                      setShowSuggestions(false);
                    }
                  }}
                  className="pl-10 bg-gray-800 border-gray-700 text-white"
                />
                {/* 动态下拉建议 */}
                {showSuggestions && searchTerm && filteredStocks.length > 0 && (
                  <div className="absolute z-50 w-full mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl max-h-96 overflow-y-auto">
                    {filteredStocks.map((stock, index) => (
                      <div
                        key={stock.code}
                        onClick={() => handleStockSelect(stock.code)}
                        className={`px-4 py-3 cursor-pointer transition-colors ${
                          index === focusedIndex 
                            ? 'bg-purple-600 text-white' 
                            : 'hover:bg-gray-700 text-gray-300'
                        } ${index !== filteredStocks.length - 1 ? 'border-b border-gray-700' : ''}`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="font-semibold">{stock.code}</span>
                            <span className="ml-3 text-gray-400">{stock.name}</span>
                          </div>
                          <span className="text-xs text-gray-500">{stock.marketName}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* 市场筛选 */}
              <Select value={marketFilter} onValueChange={(value: any) => setMarketFilter(value)}>
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                  <SelectValue placeholder="选择市场" />
                </SelectTrigger>
                <SelectContent className="bg-gray-800 border-gray-700">
                  <SelectItem value="all" className="text-white hover:bg-gray-700">全部市场</SelectItem>
                  <SelectItem value="sh" className="text-white hover:bg-gray-700">上海证券交易所</SelectItem>
                  <SelectItem value="sz" className="text-white hover:bg-gray-700">深圳证券交易所</SelectItem>
                  <SelectItem value="bj" className="text-white hover:bg-gray-700">北京证券交易所</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* 选中股票信息 */}
            {selectedStockInfo && (
              <div className="mt-4 p-4 bg-gray-800 rounded-lg border border-gray-700">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <h3 className="text-xl font-bold text-white">
                      {selectedStockInfo.code} - {selectedStockInfo.name}
                    </h3>
                    <p className="text-gray-400">{selectedStockInfo.marketName}</p>
                  </div>
                  <div className="flex gap-6">
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">买入信号</p>
                      <p className="text-2xl font-bold text-red-400">
                        {filteredKLineData.filter(d => d.signalType === 'buy').length}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">卖出信号</p>
                      <p className="text-2xl font-bold text-green-400">
                        {filteredKLineData.filter(d => d.signalType === 'sell').length}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">交易对</p>
                      <p className="text-2xl font-bold text-yellow-400">
                        {tradePairs.length}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 日期范围筛选 */}
        {selectedStock && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                日期范围筛选
              </CardTitle>
              <CardDescription>选择要查看的时间范围</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* 年月选择 */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                  {/* 年份下拉 */}
                  <div>
                    <label className="text-gray-400 text-sm mb-2 block">年份</label>
                    <Select 
                      value={selectedYear} 
                      onValueChange={(value) => {
                        setSelectedYear(value);
                        setSelectedMonth("");
                        setDateRangeType('year');
                      }}
                    >
                      <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                        <SelectValue placeholder="选择年份" />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-800 border-gray-700">
                        {availableYears.map(year => (
                          <SelectItem key={year} value={year} className="text-white hover:bg-gray-700">
                            {year}年
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* 月份下拉 */}
                  <div>
                    <label className="text-gray-400 text-sm mb-2 block">月份</label>
                    <Select 
                      value={selectedMonth} 
                      onValueChange={(value) => {
                        setSelectedMonth(value);
                        setDateRangeType('month');
                      }}
                      disabled={!selectedYear}
                    >
                      <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                        <SelectValue placeholder="选择月份" />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-800 border-gray-700">
                        {availableMonths.map(month => (
                          <SelectItem key={month} value={month} className="text-white hover:bg-gray-700">
                            {parseInt(month)}月
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* 自定义日期范围 */}
                  <div className="md:col-span-2">
                    <label className="text-gray-400 text-sm mb-2 block">自定义日期范围</label>
                    <div className="flex gap-2">
                      <Input
                        type="date"
                        value={customStartDate}
                        onChange={(e) => {
                          setCustomStartDate(e.target.value);
                          if (e.target.value && customEndDate) {
                            setDateRangeType('custom');
                          }
                        }}
                        className="bg-gray-800 border-gray-700 text-white"
                      />
                      <span className="text-gray-400 flex items-center">至</span>
                      <Input
                        type="date"
                        value={customEndDate}
                        onChange={(e) => {
                          setCustomEndDate(e.target.value);
                          if (customStartDate && e.target.value) {
                            setDateRangeType('custom');
                          }
                        }}
                        className="bg-gray-800 border-gray-700 text-white"
                      />
                    </div>
                  </div>
                </div>

                {/* 当前选择的日期范围提示 */}
                <div className="text-sm text-gray-400">
                  {dateRangeType === 'year' && selectedYear && (
                    <span>📅 当前显示：{selectedYear}年全年数据</span>
                  )}
                  {dateRangeType === 'month' && selectedYear && selectedMonth && (
                    <span>📅 当前显示：{selectedYear}年{parseInt(selectedMonth)}月数据</span>
                  )}
                  {dateRangeType === 'custom' && customStartDate && customEndDate && (
                    <span>📅 当前显示：{customStartDate} 至 {customEndDate}</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* K线图表 */}
        {selectedStock && filteredKLineData.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white">K线价格图</CardTitle>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={signalFilter === 'all' ? 'default' : 'outline'}
                    onClick={() => setSignalFilter('all')}
                    className={signalFilter === 'all' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                  >
                    全部信号
                  </Button>
                  <Button
                    size="sm"
                    variant={signalFilter === 'buy' ? 'default' : 'outline'}
                    onClick={() => setSignalFilter('buy')}
                    className={signalFilter === 'buy' ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                  >
                    仅买入⚪
                  </Button>
                  <Button
                    size="sm"
                    variant={signalFilter === 'sell' ? 'default' : 'outline'}
                    onClick={() => setSignalFilter('sell')}
                    className={signalFilter === 'sell' ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                  >
                    仅卖出▲
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* K线价格图 */}
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={filteredKLineData} syncId="stockChart">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    tickFormatter={(value) => value.substring(5)}
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend 
                    wrapperStyle={{ color: '#9ca3af' }}
                    iconType="line"
                  />
                  
                  {/* 收盘价折线图 */}
                  <Line 
                    type="monotone"
                    dataKey="close"
                    stroke="#a855f7"
                    strokeWidth={2}
                    dot={false}
                    name="收盘价"
                    isAnimationActive={false}
                  />

                  {/* 买入信号（红色圆圈） */}
                  {(signalFilter === 'all' || signalFilter === 'buy') && (
                    <Scatter
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'buy')}
                      fill="#ef4444"
                      shape="circle"
                      r={7}
                      name="买入信号"
                    />
                  )}

                  {/* 卖出信号（绿色三角） */}
                  {(signalFilter === 'all' || signalFilter === 'sell') && (
                    <Scatter
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'sell')}
                      fill="#22c55e"
                      shape="triangle"
                      r={9}
                      name="卖出信号"
                    />
                  )}

                  {/* 交易对虚线 */}
                  {signalFilter === 'all' && tradePairs.map((pair, index) => (
                    <ReferenceLine
                      key={`pair-${index}`}
                      segment={[
                        { x: pair.buy.date, y: pair.buy.close },
                        { x: pair.sell.date, y: pair.sell.close }
                      ]}
                      stroke={parseFloat(pair.profitPercent) >= 0 ? '#ef4444' : '#22c55e'}
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{
                        value: `${pair.profitPercent}%`,
                        position: 'center',
                        fill: parseFloat(pair.profitPercent) >= 0 ? '#ef4444' : '#22c55e',
                        fontSize: 12,
                        fontWeight: 'bold'
                      }}
                    />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>

              {/* 成交量图 */}
              <div className="mt-6">
                <h3 className="text-white font-semibold mb-2">成交量</h3>
                <ResponsiveContainer width="100%" height={150}>
                  <ComposedChart data={filteredKLineData} syncId="stockChart">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      tickFormatter={(value) => value.substring(5)}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      tickFormatter={(value) => `${(value / 10000).toFixed(0)}万`}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                      itemStyle={{ color: '#9ca3af' }}
                      formatter={(value: any) => [`${(value / 10000).toFixed(0)}万`, '成交量']}
                    />
                    <Bar dataKey="volume" fill="#6b7280" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 交易对统计 */}
        {selectedStock && tradePairs.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">交易对统计</CardTitle>
              <CardDescription>基于FIFO算法匹配的买卖交易对</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">序号</th>
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">买入日期</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">买入价格</th>
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">卖出日期</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">卖出价格</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">交易金额</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">盈亏</th>
                      <th className="text-right py-3 px-4 text-gray-400 font-medium">收益率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tradePairs.map((pair, index) => {
                      const isProfit = parseFloat(pair.profitPercent) >= 0;
                      return (
                        <tr key={index} className="border-b border-gray-800 hover:bg-gray-800 transition-colors">
                          <td className="py-3 px-4 text-gray-300">{index + 1}</td>
                          <td className="py-3 px-4 text-gray-300">{pair.buy.date}</td>
                          <td className="py-3 px-4 text-right text-gray-300">¥{pair.buy.close.toFixed(2)}</td>
                          <td className="py-3 px-4 text-gray-300">{pair.sell.date}</td>
                          <td className="py-3 px-4 text-right text-gray-300">¥{pair.sell.close.toFixed(2)}</td>
                          <td className="py-3 px-4 text-right text-gray-300">¥{pair.amount.toFixed(2)}</td>
                          <td className={`py-3 px-4 text-right font-semibold ${isProfit ? 'text-red-400' : 'text-green-400'}`}>
                            {isProfit ? '+' : ''}¥{pair.profit.toFixed(2)}
                          </td>
                          <td className={`py-3 px-4 text-right font-semibold ${isProfit ? 'text-red-400' : 'text-green-400'}`}>
                            {isProfit ? '+' : ''}{pair.profitPercent}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                  <tfoot>
                    <tr className="border-t-2 border-gray-700 bg-gray-800">
                      <td colSpan={5} className="py-3 px-4 text-right text-gray-400 font-medium">总计：</td>
                      <td className="py-3 px-4 text-right text-white font-bold">
                        ¥{tradePairs.reduce((sum, pair) => sum + pair.amount, 0).toFixed(2)}
                      </td>
                      <td className={`py-3 px-4 text-right font-bold ${
                        tradePairs.reduce((sum, pair) => sum + pair.profit, 0) >= 0 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {tradePairs.reduce((sum, pair) => sum + pair.profit, 0) >= 0 ? '+' : ''}
                        ¥{tradePairs.reduce((sum, pair) => sum + pair.profit, 0).toFixed(2)}
                      </td>
                      <td className={`py-3 px-4 text-right font-bold ${
                        tradePairs.reduce((sum, pair) => sum + parseFloat(pair.profitPercent), 0) >= 0 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {tradePairs.reduce((sum, pair) => sum + parseFloat(pair.profitPercent), 0) >= 0 ? '+' : ''}
                        {(tradePairs.reduce((sum, pair) => sum + parseFloat(pair.profitPercent), 0) / tradePairs.length).toFixed(2)}%
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 空状态提示 */}
        {!selectedStock && (
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="py-12">
              <div className="text-center text-gray-500">
                <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">请先搜索并选择一只股票</p>
                <p className="text-sm mt-2">在上方搜索框输入股票代码或名称</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
