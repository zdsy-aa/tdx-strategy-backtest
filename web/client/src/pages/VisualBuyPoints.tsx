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

export default function VisualBuyPoints() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStock, setSelectedStock] = useState<string>("");
  const [marketFilter, setMarketFilter] = useState<"all" | "sh" | "sz" | "bj">("all");
  const [signalFilter, setSignalFilter] = useState<"all" | "buy" | "sell">("all");
  const [timeRange, setTimeRange] = useState<"1m" | "3m" | "6m" | "1y" | "all">("3m");

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
      .slice(0, 100); // 限制显示前100个
  }, [stockReports, searchTerm, marketFilter]);

  // 模拟K线数据（实际应该从后端API获取）
  const generateMockKLineData = (stockCode: string): KLineData[] => {
    const data: KLineData[] = [];
    let basePrice = 10 + Math.random() * 20;
    const startDate = new Date('2025-10-01');
    
    for (let i = 0; i < 60; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      const change = (Math.random() - 0.5) * 2;
      const open = basePrice;
      const close = basePrice + change;
      const high = Math.max(open, close) + Math.random() * 1;
      const low = Math.min(open, close) - Math.random() * 1;
      const volume = Math.floor(Math.random() * 1000000) + 100000;
      
      // 随机生成一些买卖信号
      let signal: string | undefined;
      let signalType: 'buy' | 'sell' | undefined;
      
      if (Math.random() > 0.9) {
        if (Math.random() > 0.5) {
          signal = "六脉6红";
          signalType = "buy";
        } else {
          signal = "买点2";
          signalType = "buy";
        }
      } else if (Math.random() > 0.95) {
        signal = "卖点1";
        signalType = "sell";
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

  // 根据时间范围和信号类型筛选K线数据
  const filteredKLineData = useMemo(() => {
    let data = klineData;
    
    // 时间范围筛选
    if (timeRange !== "all" && data.length > 0) {
      const days = {
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365
      }[timeRange] || 90;
      
      data = data.slice(-days);
    }
    
    // 信号类型筛选（不过滤数据，只影响显示）
    return data;
  }, [klineData, timeRange]);

  // 获取选中股票的信息
  const selectedStockInfo = useMemo(() => {
    return stockReports.find(stock => stock.code === selectedStock);
  }, [stockReports, selectedStock]);

  // 自定义K线形状
  const CustomCandlestick = (props: any) => {
    const { x, y, width, height, payload } = props;
    const { open, close, high, low } = payload;
    
    const isUp = close > open;
    const color = isUp ? "#ef4444" : "#22c55e"; // 涨红跌绿
    
    const bodyHeight = Math.abs(close - open);
    const bodyY = Math.min(close, open);
    
    return (
      <g>
        {/* 上影线 */}
        <line
          x1={x + width / 2}
          y1={y + (high - Math.max(open, close))}
          x2={x + width / 2}
          y2={y + (high - high)}
          stroke={color}
          strokeWidth={1}
        />
        {/* 下影线 */}
        <line
          x1={x + width / 2}
          y1={y + (high - Math.min(open, close))}
          x2={x + width / 2}
          y2={y + (high - low)}
          stroke={color}
          strokeWidth={1}
        />
        {/* K线实体 */}
        <rect
          x={x}
          y={y + (high - bodyY)}
          width={width}
          height={bodyHeight || 1}
          fill={color}
          stroke={color}
        />
      </g>
    );
  };

  // 自定义Tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isUp = data.close > data.open;
      const change = ((data.close - data.open) / data.open * 100).toFixed(2);
      
      return (
        <div className="bg-gray-900 border border-gray-700 p-3 rounded-lg shadow-lg">
          <p className="text-white font-semibold mb-2">{data.date}</p>
          <div className="space-y-1 text-sm">
            <p className="text-gray-300">开盘: <span className="text-white">{data.open}</span></p>
            <p className="text-gray-300">收盘: <span className={isUp ? "text-red-400" : "text-green-400"}>{data.close}</span></p>
            <p className="text-gray-300">最高: <span className="text-white">{data.high}</span></p>
            <p className="text-gray-300">最低: <span className="text-white">{data.low}</span></p>
            <p className="text-gray-300">涨跌幅: <span className={isUp ? "text-red-400" : "text-green-400"}>{change}%</span></p>
            <p className="text-gray-300">成交量: <span className="text-white">{(data.volume / 10000).toFixed(2)}万</span></p>
            {data.signal && (
              <p className="text-yellow-400 font-semibold mt-2">
                {data.signalType === 'buy' ? '🔵 ' : '🔴 '}
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
              {/* 搜索框 */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="搜索股票代码或名称"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                  }
                }}
                className="pl-10 bg-gray-800 border-gray-700 text-white"
              />
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
              <CardDescription>红色为涨，绿色为跌 | 蓝点为买入信号，红点为卖出信号</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={500}>
                <ComposedChart data={filteredKLineData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9ca3af"
                    tick={{ fill: '#9ca3af' }}
                    tickFormatter={(value) => value.slice(5)} // 只显示月-日
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
                  
                  {/* 买入信号标注 */}
                  {(signalFilter === "all" || signalFilter === "buy") && (
                    <Scatter
                      yAxisId="price"
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'buy')}
                      fill="#3b82f6"
                      shape="circle"
                      name="买入信号"
                      r={6}
                    />
                  )}
                  
                  {/* 卖出信号标注 */}
                  {(signalFilter === "all" || signalFilter === "sell") && (
                    <Scatter
                      yAxisId="price"
                      dataKey="close"
                      data={filteredKLineData.filter(d => d.signalType === 'sell')}
                      fill="#ef4444"
                      shape="circle"
                      name="卖出信号"
                      r={6}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* 未选择股票提示 */}
        {!selectedStock && (
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="py-20">
              <div className="text-center text-gray-400">
                <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-xl">请选择股票查看K线图</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 信号说明 */}
        <Card className="bg-gray-900 border-gray-800">
          <CardHeader>
            <CardTitle className="text-white">信号说明</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-blue-400 font-semibold flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  买入信号
                </h4>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• <span className="text-yellow-400">六脉6红</span>: 六个指标同时看多，强烈买入信号</li>
                  <li>• <span className="text-yellow-400">六脉5红</span>: 五个指标看多，较强买入信号</li>
                  <li>• <span className="text-yellow-400">买点1</span>: 吸筹指标上穿14，庄家建仓信号</li>
                  <li>• <span className="text-yellow-400">买点2</span>: 庄家线上穿散户线，主力拉升信号</li>
                  <li>• <span className="text-yellow-400">缠论一买</span>: 底分型+下跌趋势，抄底信号</li>
                  <li>• <span className="text-yellow-400">缠论二买</span>: 回踩不破前低，确认上涨信号</li>
                  <li>• <span className="text-yellow-400">缠论三买</span>: 回踩不破中枢，追涨信号</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="text-red-400 font-semibold flex items-center gap-2">
                  <TrendingDown className="w-4 h-4" />
                  卖出信号
                </h4>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• <span className="text-green-400">卖点1</span>: 庄家线高位回落，主力出货信号</li>
                  <li>• <span className="text-green-400">卖点2</span>: 散户线上穿庄家线，散户接盘信号</li>
                  <li>• <span className="text-green-400">缠论一卖</span>: 顶分型+上涨趋势，逃顶信号</li>
                  <li>• <span className="text-green-400">缠论二卖</span>: 反弹不过前高，确认下跌信号</li>
                  <li>• <span className="text-green-400">缠论三卖</span>: 反弹不过中枢，杀跌信号</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
