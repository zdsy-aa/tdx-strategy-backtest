import { useState, useMemo } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Scatter } from 'recharts';
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
}

type DateRangeType = 'all' | '1year' | '6months' | '3months' | '1month' | 'custom';

export default function VisualBuyPoints() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStock, setSelectedStock] = useState<string>("");
  const [marketFilter, setMarketFilter] = useState<"all" | "sh" | "sz" | "bj">("all");
  const [signalFilter, setSignalFilter] = useState<"all" | "buy" | "sell">("all");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [dateRangeType, setDateRangeType] = useState<DateRangeType>('6months');
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

  // 模拟K线数据生成
  const generateMockKLineData = (stockCode: string): KLineData[] => {
    const data: KLineData[] = [];
    let basePrice = 10 + Math.random() * 20;
    const startDate = new Date('2024-01-01');
    let lastSignalIndex = -10;
    let lastSignalType: 'buy' | 'sell' | undefined;
    
    for (let i = 0; i < 365; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      const change = (Math.random() - 0.5) * 2;
      const open = basePrice;
      const close = basePrice + change;
      const high = Math.max(open, close) + Math.random() * 1;
      const low = Math.min(open, close) - Math.random() * 1;
      const volume = Math.floor(Math.random() * 1000000) + 100000;
      
      let signal: string | undefined;
      let signalType: 'buy' | 'sell' | undefined;
      
      // 确保买卖信号交替出现，且有足够间隔
      if (i - lastSignalIndex >= 5 && Math.random() > 0.88) {
        if (lastSignalType === 'buy') {
          signal = "卖点１";
          signalType = "sell";
        } else {
          const buySignals = ["六脉６红", "买点２", "缠论一买"];
          signal = buySignals[Math.floor(Math.random() * buySignals.length)];
          signalType = "buy";
        }
        lastSignalIndex = i;
        lastSignalType = signalType;
      }
      
      data.push({
        date: date.toISOString().split('T')[0],
        open: parseFloat(open.toFixed(2)),
        high: parseFloat(high.toFixed(2)),
        low: parseFloat(low.toFixed(2)),
        close: parseFloat(close.toFixed(2)),
        volume,
        signal,
        signalType
      });
      
      basePrice = close;
    }
    
    return data;
  };

  // 获取选中股票的K线数据
  const klineData = useMemo(() => {
    if (!selectedStock) return [];
    return generateMockKLineData(selectedStock);
  }, [selectedStock]);

  // 根据日期范围筛选数据
  const filteredKLineData = useMemo(() => {
    if (klineData.length === 0) return [];
    
    const lastDate = new Date(klineData[klineData.length - 1].date);
    let startDate: Date;
    
    switch (dateRangeType) {
      case '1month':
        startDate = new Date(lastDate);
        startDate.setMonth(startDate.getMonth() - 1);
        break;
      case '3months':
        startDate = new Date(lastDate);
        startDate.setMonth(startDate.getMonth() - 3);
        break;
      case '6months':
        startDate = new Date(lastDate);
        startDate.setMonth(startDate.getMonth() - 6);
        break;
      case '1year':
        startDate = new Date(lastDate);
        startDate.setFullYear(startDate.getFullYear() - 1);
        break;
      case 'custom':
        if (customStartDate && customEndDate) {
          return klineData.filter(d => d.date >= customStartDate && d.date <= customEndDate);
        }
        return klineData;
      case 'all':
      default:
        return klineData;
    }
    
    return klineData.filter(d => new Date(d.date) >= startDate);
  }, [klineData, dateRangeType, customStartDate, customEndDate]);

  // 买卖匹配逻辑 (FIFO)
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
          pairs.push({ 
            buy: buyDay, 
            sell: day,
            profit,
            profitPercent
          });
        }
      }
    });
    
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
                  <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-lg z-50 max-h-[400px] overflow-y-auto">
                    {filteredStocks.map((stock, index) => (
                      <div
                        key={stock.code}
                        className={`px-4 py-3 cursor-pointer hover:bg-gray-700 transition-colors ${
                          index === focusedIndex ? 'bg-gray-700' : ''
                        }`}
                        onClick={() => handleStockSelect(stock.code)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-white font-medium">{stock.code}</div>
                            <div className="text-gray-400 text-sm">{stock.name}</div>
                          </div>
                          <div className="text-xs text-gray-500">{stock.marketName}</div>
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
                <SelectContent>
                  <SelectItem value="all">全部市场</SelectItem>
                  <SelectItem value="sh">沪市</SelectItem>
                  <SelectItem value="sz">深市</SelectItem>
                  <SelectItem value="bj">北交所</SelectItem>
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
              <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                <Button
                  onClick={() => setDateRangeType('1month')}
                  variant={dateRangeType === '1month' ? 'default' : 'outline'}
                  className={dateRangeType === '1month' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  1个月
                </Button>
                <Button
                  onClick={() => setDateRangeType('3months')}
                  variant={dateRangeType === '3months' ? 'default' : 'outline'}
                  className={dateRangeType === '3months' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  3个月
                </Button>
                <Button
                  onClick={() => setDateRangeType('6months')}
                  variant={dateRangeType === '6months' ? 'default' : 'outline'}
                  className={dateRangeType === '6months' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  6个月
                </Button>
                <Button
                  onClick={() => setDateRangeType('1year')}
                  variant={dateRangeType === '1year' ? 'default' : 'outline'}
                  className={dateRangeType === '1year' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  1年
                </Button>
                <Button
                  onClick={() => setDateRangeType('all')}
                  variant={dateRangeType === 'all' ? 'default' : 'outline'}
                  className={dateRangeType === 'all' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  全部
                </Button>
                <Button
                  onClick={() => setDateRangeType('custom')}
                  variant={dateRangeType === 'custom' ? 'default' : 'outline'}
                  className={dateRangeType === 'custom' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-800 border-gray-700 text-white hover:bg-gray-700'}
                >
                  自定义
                </Button>
              </div>
              
              {/* 自定义日期选择 */}
              {dateRangeType === 'custom' && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-gray-400 text-sm mb-2 block">开始日期</label>
                    <Input
                      type="date"
                      value={customStartDate}
                      onChange={(e) => setCustomStartDate(e.target.value)}
                      className="bg-gray-800 border-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="text-gray-400 text-sm mb-2 block">结束日期</label>
                    <Input
                      type="date"
                      value={customEndDate}
                      onChange={(e) => setCustomEndDate(e.target.value)}
                      className="bg-gray-800 border-gray-700 text-white"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* K线图区域 */}
        {selectedStock && filteredKLineData.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">K线图与信号标注</CardTitle>
              <CardDescription>
                红色⚪：买入信号 | 绿色▲：卖出信号 | 虚线：交易路径与收益率
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* K线价格图 */}
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart 
                  data={filteredKLineData} 
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  syncId="stockChart"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    tickFormatter={(value) => typeof value === 'string' ? value.slice(5) : String(value)}
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    domain={['auto', 'auto']}
                    label={{ value: '价格', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  
                  {/* 收盘价线 */}
                  <Line 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#8b5cf6" 
                    strokeWidth={2}
                    dot={false}
                    name="收盘价"
                  />
                  
                  {/* 买卖连线 */}
                  {tradePairs.map((pair, index) => (
                    <ReferenceLine
                      key={`trade-${index}`}
                      segment={[
                        { x: pair.buy.date, y: pair.buy.close },
                        { x: pair.sell.date, y: pair.sell.close }
                      ]}
                      stroke={parseFloat(pair.profitPercent) >= 0 ? '#ef4444' : '#22c55e'}
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{
                        value: `${pair.profitPercent}%`,
                        position: 'top',
                        fill: parseFloat(pair.profitPercent) >= 0 ? '#ef4444' : '#22c55e',
                        fontSize: 12,
                        fontWeight: 'bold',
                        offset: 10
                      }}
                    />
                  ))}
                  
                  {/* 买入信号标注（红色圆圈） */}
                  {(signalFilter === "all" || signalFilter === "buy") && (
                    <Scatter
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'buy')}
                      fill="#ef4444"
                      shape="circle"
                      name="买入信号"
                      r={7}
                    />
                  )}
                  
                  {/* 卖出信号标注（绿色三角） */}
                  {(signalFilter === "all" || signalFilter === "sell") && (
                    <Scatter
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'sell')}
                      fill="#22c55e"
                      shape="triangle"
                      name="卖出信号"
                      r={9}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>

              {/* 成交量图 */}
              <ResponsiveContainer width="100%" height={150}>
                <ComposedChart 
                  data={filteredKLineData} 
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  syncId="stockChart"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    tickFormatter={(value) => typeof value === 'string' ? value.slice(5) : String(value)}
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    tickFormatter={(value) => `${(value / 10000).toFixed(0)}万`}
                    label={{ value: '成交量', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  
                  {/* 成交量柱状图 */}
                  <Bar 
                    dataKey="volume" 
                    fill="#4b5563" 
                    opacity={0.6}
                    name="成交量"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* 交易对统计 */}
        {tradePairs.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">交易对统计（当前显示范围）</CardTitle>
              <CardDescription>买卖信号匹配结果（FIFO）</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {tradePairs.map((pair, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors">
                    <div className="flex-1">
                      <p className="text-white font-semibold">
                        交易 #{index + 1}
                      </p>
                      <p className="text-gray-400 text-sm">
                        买入: {pair.buy.date} @ ¥{pair.buy.close} | 卖出: {pair.sell.date} @ ¥{pair.sell.close}
                      </p>
                    </div>
                    <div className={`text-right font-bold ${parseFloat(pair.profitPercent) >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                      <p className="text-lg">{pair.profitPercent}%</p>
                      <p className="text-sm">{pair.profit > 0 ? '+' : ''}¥{pair.profit}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* 未选择股票提示 */}
        {!selectedStock && (
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="py-20">
              <div className="text-center text-gray-400">
                <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">请在上方搜索框输入股票代码或名称</p>
                <p className="text-sm mt-2">支持模糊搜索，选中后自动加载K线图</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
