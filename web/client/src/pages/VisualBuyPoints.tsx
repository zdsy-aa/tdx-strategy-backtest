import { useState, useMemo } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Search, TrendingUp, TrendingDown } from "lucide-react";
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

export default function VisualBuyPoints() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStock, setSelectedStock] = useState<string>("");
  const [marketFilter, setMarketFilter] = useState<"all" | "sh" | "sz" | "bj">("all");
  const [signalFilter, setSignalFilter] = useState<"all" | "buy" | "sell">("all");
  const [timeRange, setTimeRange] = useState<"1m" | "3m" | "6m" | "1y" | "all">("3m");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);

  // 获取股票列表
  const stockReports: StockReport[] = useMemo(() => {
    return stockReportsData as StockReport[];
  }, []);

  // 筛选股票列表
  const filteredStocks = useMemo(() => {
    return stockReports
      .filter(stock => {
        const matchSearch = stock.code.includes(searchTerm) || 
          stock.name.toLowerCase().includes(searchTerm.toLowerCase());
        const matchMarket = marketFilter === "all" || stock.market === marketFilter;
        return matchSearch && matchMarket;
      })
      .slice(0, 100);
  }, [stockReports, searchTerm, marketFilter]);

  // 模拟K线数据
  const generateMockKLineData = (stockCode: string): KLineData[] => {
    const data: KLineData[] = [];
    let basePrice = 10 + Math.random() * 20;
    const startDate = new Date('2025-10-01');
    let lastSignalIndex = -10;
    let lastSignalType: 'buy' | 'sell' | undefined;
    
    for (let i = 0; i < 60; i++) {
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
      
      if (i - lastSignalIndex >= 3 && Math.random() > 0.85) {
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

  // 买卖匹配逻辑 (FIFO)
  const tradePairs = useMemo(() => {
    const pairs: TradePair[] = [];
    const buyQueue: KLineData[] = [];
    
    klineData.forEach(day => {
      if (day.signalType === 'buy') {
        buyQueue.push(day);
      } else if (day.signalType === 'sell') {
        if (buyQueue.length > 0) {
          const buyDay = buyQueue.shift()!;
          const profit = parseFloat((sellPrice - buyDay.close).toFixed(2));
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
  }, [klineData]);

  // 根据时间范围筛选K线数据
  const filteredKLineData = useMemo(() => {
    let data = klineData;
    
    if (timeRange !== "all" && data.length > 0) {
      const days = {
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365
      }[timeRange];
      
      if (days) {
        const lastDate = new Date(data[data.length - 1].date);
        const cutoffDate = new Date(lastDate);
        cutoffDate.setDate(cutoffDate.getDate() - days);
        
        data = data.filter(d => new Date(d.date) >= cutoffDate);
      }
    }
    
    return data;
  }, [klineData, timeRange]);

  // 筛选在当前时间范围内的交易对
  const filteredTradePairs = useMemo(() => {
    if (filteredKLineData.length === 0) return [];
    const firstDate = new Date(filteredKLineData[0].date);
    return tradePairs.filter(pair => new Date(pair.sell.date) >= firstDate);
  }, [tradePairs, filteredKLineData]);

  // 获取选中股票的信息
  const selectedStockInfo = useMemo(() => {
    return stockReports.find(stock => stock.code === selectedStock);
  }, [stockReports, selectedStock]);

  // 自定义Tooltip
  const CustomTooltip = (props: any) => {
    const { active, payload } = props;
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
          <p className="text-white font-semibold">{data.date}</p>
          <div className="text-sm space-y-1">
            <p className="text-gray-300">开: {data.open}</p>
            <p className="text-gray-300">收: {data.close}</p>
            <p className="text-gray-300">高: {data.high}</p>
            <p className="text-gray-300">低: {data.low}</p>
            {data.signal && (
              <p className="text-yellow-400 font-semibold">
                {data.signal}
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
            <p className="text-gray-400">K线图展示与信号标注</p>
          </div>
        </div>

        {/* 股票筛选区域 */}
        <Card className="bg-gray-900 border-gray-800">
          <CardHeader>
            <CardTitle className="text-white">股票筛选</CardTitle>
            <CardDescription>选择股票查看K线图和买卖点信号</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {/* 搜索框（带动态下拉） */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 z-10" />
                <Input
                  placeholder="搜索股票代码或名称"
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
                        setSelectedStock(filteredStocks[focusedIndex].code);
                        setSearchTerm('');
                        setShowSuggestions(false);
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
                  <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-lg z-50 max-h-[300px] overflow-y-auto">
                    {filteredStocks.slice(0, 10).map((stock, index) => (
                      <div
                        key={stock.code}
                        className={`px-4 py-2 cursor-pointer hover:bg-gray-700 ${
                          index === focusedIndex ? 'bg-gray-700' : ''
                        }`}
                        onClick={() => {
                          setSelectedStock(stock.code);
                          setSearchTerm('');
                          setShowSuggestions(false);
                        }}
                      >
                        <div className="text-white font-medium">{stock.code} - {stock.name}</div>
                        <div className="text-gray-400 text-sm">{stock.marketName}</div>
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

              {/* 股票选择 */}
              <Select value={selectedStock} onValueChange={setSelectedStock}>
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                  <SelectValue placeholder="选择股票" />
                </SelectTrigger>
                <SelectContent className="max-h-[300px]">
                  {filteredStocks.map((stock) => (
                    <SelectItem key={stock.code} value={stock.code}>
                      {stock.code} - {stock.name} ({stock.marketName})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* 信号筛选 */}
              <Select value={signalFilter} onValueChange={(value: any) => setSignalFilter(value)}>
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                  <SelectValue placeholder="信号类型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部信号</SelectItem>
                  <SelectItem value="buy">买入信号</SelectItem>
                  <SelectItem value="sell">卖出信号</SelectItem>
                </SelectContent>
              </Select>

              {/* 时间范围筛选 */}
              <Select value={timeRange} onValueChange={(value: any) => setTimeRange(value)}>
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                  <SelectValue placeholder="时间范围" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1m">1个月</SelectItem>
                  <SelectItem value="3m">3个月</SelectItem>
                  <SelectItem value="6m">6个月</SelectItem>
                  <SelectItem value="1y">1年</SelectItem>
                  <SelectItem value="all">全部</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {selectedStockInfo && (
              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold text-white">
                      {selectedStockInfo.code} - {selectedStockInfo.name}
                    </h3>
                    <p className="text-gray-400">{selectedStockInfo.marketName}</p>
                  </div>
                  <div className="flex gap-4">
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">买入信号</p>
                      <p className="text-2xl font-bold text-blue-400">
                        {klineData.filter(d => d.signalType === 'buy').length}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">卖出信号</p>
                      <p className="text-2xl font-bold text-red-400">
                        {klineData.filter(d => d.signalType === 'sell').length}
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-400 text-sm">交易对</p>
                      <p className="text-2xl font-bold text-green-400">
                        {filteredTradePairs.length}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* K线图区域 */}
        {selectedStock && filteredKLineData.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">K线图与信号标注</CardTitle>
              <CardDescription>红色为涨，绿色为跌 | 蓝色上三角：买入 | 红色下三角：卖出 | 虚线：买卖连线</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={500}>
                <ComposedChart data={filteredKLineData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af' }}
                    tickFormatter={(value) => value.slice(5)}
                  />
                  <YAxis 
                    yAxisId="price"
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af' }}
                    domain={['dataMin - 1', 'dataMax + 1']}
                  />
                  <YAxis 
                    yAxisId="volume"
                    orientation="right"
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af' }}
                    tickFormatter={(value) => `${(value / 10000).toFixed(0)}万`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  
                  {/* 成交量柱状图 */}
                  <Bar 
                    yAxisId="volume"
                    dataKey="volume" 
                    fill="#4b5563" 
                    opacity={0.3}
                    name="成交量"
                  />
                  
                  {/* 收盘价线 */}
                  <Line 
                    yAxisId="price"
                    type="monotone" 
                    dataKey="close" 
                    stroke="#8b5cf6" 
                    strokeWidth={2}
                    dot={false}
                    name="收盘价"
                  />
                  
                  {/* 买卖连线 */}
                  {filteredTradePairs.map((pair, index) => (
                    <ReferenceLine
                      key={`trade-${index}`}
                      x1={pair.buy.date}
                      x2={pair.sell.date}
                      y1={pair.buy.close}
                      y2={pair.sell.close}
                      yAxisId="price"
                      stroke={parseFloat(pair.profitPercent) >= 0 ? '#3b82f6' : '#ef4444'}
                      strokeDasharray="5 5"
                      strokeWidth={1.5}
                      label={{
                        value: `${pair.profitPercent}%`,
                        position: 'top',
                        fill: parseFloat(pair.profitPercent) >= 0 ? '#3b82f6' : '#ef4444',
                        fontSize: 11,
                        offset: 5
                      }}
                    />
                  ))}
                  
                  {/* 买入信号标注 */}
                  {(signalFilter === "all" || signalFilter === "buy") && (
                    <Scatter
                      yAxisId="price"
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'buy')}
                      fill="#3b82f6"
                      shape="triangle"
                      name="买入信号"
                      r={8}
                    />
                  )}
                  
                  {/* 卖出信号标注 */}
                  {(signalFilter === "all" || signalFilter === "sell") && (
                    <Scatter
                      yAxisId="price"
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'sell').map(d => ({
                        ...d,
                        close: d.close * 1.02
                      }))}
                      fill="#ef4444"
                      shape={(props: any) => {
                        const { cx, cy } = props;
                        return (
                          <polygon
                            points={`${cx},${cy + 8} ${cx - 8},${cy - 8} ${cx + 8},${cy - 8}`}
                            fill="#ef4444"
                            stroke="#fff"
                            strokeWidth={1}
                          />
                        );
                      }}
                      name="卖出信号"
                      r={8}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* 交易对统计 */}
        {filteredTradePairs.length > 0 && (
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">交易对统计</CardTitle>
              <CardDescription>买卖信号匹配结果（FIFO）</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {filteredTradePairs.map((pair, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                    <div className="flex-1">
                      <p className="text-white font-semibold">
                        交易 #{index + 1}
                      </p>
                      <p className="text-gray-400 text-sm">
                        买入: {pair.buy.date} @ {pair.buy.close} | 卖出: {pair.sell.date} @ {pair.sell.close}
                      </p>
                    </div>
                    <div className={`text-right font-bold ${parseFloat(pair.profitPercent) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      <p>{pair.profitPercent}%</p>
                      <p className="text-sm">{pair.profit > 0 ? '+' : ''}{pair.profit}</p>
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
                <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>请选择股票查看K线图</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
