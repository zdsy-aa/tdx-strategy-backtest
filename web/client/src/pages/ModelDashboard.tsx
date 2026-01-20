"use client";

import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, TrendingUp, Activity, Search, Info, Zap, Target, BarChart3, ChevronDown, ChevronUp } from "lucide-react";
import Layout from "@/components/Layout";
import dashboardDataRaw from "@/data/dashboard.json";
import { useState, useMemo } from "react";

interface DashboardData {
  generated_at: string;
  markets: {
    [key: string]: {
      total: number;
      ok: number;
      fail: number;
    };
  };
  counts: {
    symbols_total: number;
    symbols_ok: number;
    symbols_fail: number;
  };
  top: Array<{
    market: string;
    code: string;
    name: string;
    last_date: string;
    final_score: number;
    score_A: number;
    score_B: number;
    score_C?: number;
    signals_count: number;
  }>;
}

// 策略说明常量
const STRATEGY_INFO = {
  A: {
    name: "趋势跟踪策略 (MA交叉)",
    shortName: "MA交叉",
    description: "基于均线系统的趋势跟踪策略，当短期均线(MA5)上穿长期均线(MA20)时产生买入信号",
    signals: [
      { type: "A_MA_CROSS_UP", name: "金叉信号", desc: "MA5上穿MA20，表示短期趋势转强" }
    ],
    scoreRange: "0-60分",
    color: "#3b82f6"
  },
  B: {
    name: "超卖反弹策略 (RSI+量能)",
    shortName: "RSI超卖",
    description: "结合RSI超卖指标和成交量放大信号，捕捉超跌反弹机会",
    signals: [
      { type: "B_RSI_OVERSOLD", name: "RSI超卖", desc: "RSI14低于30，表示股价可能超卖" },
      { type: "B_VOLUME_SPIKE", name: "放量信号", desc: "成交量是5日均量的1.5倍以上" }
    ],
    scoreRange: "0-70分",
    color: "#ef4444"
  },
  C: {
    name: "高级指标共振策略 (缠论+六脉神剑+摇钱树+买卖点)",
    shortName: "高级指标",
    description: "融合缠论笔结构、六脉神剑共振、黄金摇钱树、庄家散户线等高级技术指标的综合选股系统",
    signals: [
      { type: "C_CHAN_BUY", name: "缠论买点", desc: "缠论一买/二买/三买信号，最高优先级", priority: "极高" },
      { type: "C_SIX_VEINS", name: "六脉神剑", desc: "六个技术指标同时共振，形成强买入信号", priority: "高" },
      { type: "C_MONEY_TREE", name: "黄金摇钱树", desc: "三重过滤条件同时满足的选股信号", priority: "中" },
      { type: "C_BANKER_CROSS", name: "庄家上穿", desc: "庄家线上穿散户线，主力建仓信号", priority: "中" }
    ],
    scoreRange: "0-80分",
    color: "#8b5cf6"
  }
};

export default function ModelDashboard() {
  const dashboardData = dashboardDataRaw as unknown as DashboardData;
  const [searchTerm, setSearchTerm] = useState("");
  const [marketFilter, setMarketFilter] = useState("all");
  const [scoreFilter, setScoreFilter] = useState("all");
  const [selectedStock, setSelectedStock] = useState<typeof dashboardData.top[0] | null>(null);
  const [expandedStrategies, setExpandedStrategies] = useState<Set<string>>(new Set(["A", "B", "C"]));

  // 过滤数据
  const filteredData = useMemo(() => {
    if (!dashboardData?.top) return [];
    return dashboardData.top.filter(item => {
      const matchesSearch = item.code.includes(searchTerm) || item.name.includes(searchTerm);
      const matchesMarket = marketFilter === "all" || item.market === marketFilter;
      const matchesScore = scoreFilter === "all" || 
        (scoreFilter === "high" && item.final_score >= 100) ||
        (scoreFilter === "medium" && item.final_score >= 50 && item.final_score < 100) ||
        (scoreFilter === "low" && item.final_score < 50 && item.final_score > 0) ||
        (scoreFilter === "none" && item.final_score === 0);
      return matchesSearch && matchesMarket && matchesScore;
    });
  }, [dashboardData, searchTerm, marketFilter, scoreFilter]);

  // 统计有信号的股票
  const stocksWithSignals = useMemo(() => {
    if (!dashboardData?.top) return { total: 0, strategyA: 0, strategyB: 0, strategyC: 0, dual: 0, triple: 0 };
    const withA = dashboardData.top.filter(s => s.score_A > 0).length;
    const withB = dashboardData.top.filter(s => s.score_B > 0).length;
    const withC = dashboardData.top.filter(s => (s.score_C ?? 0) > 0).length;
    const withAny = dashboardData.top.filter(s => s.final_score > 0).length;
    const withDual = dashboardData.top.filter(s => {
      const count = (s.score_A > 0 ? 1 : 0) + (s.score_B > 0 ? 1 : 0) + ((s.score_C ?? 0) > 0 ? 1 : 0);
      return count >= 2;
    }).length;
    const withTriple = dashboardData.top.filter(s => s.score_A > 0 && s.score_B > 0 && (s.score_C ?? 0) > 0).length;
    return { total: withAny, strategyA: withA, strategyB: withB, strategyC: withC, dual: withDual, triple: withTriple };
  }, [dashboardData]);

  // 信号强度分布
  const scoreDistribution = useMemo(() => {
    if (!dashboardData?.top) return [];
    const ranges = [
      { range: "150+", min: 150, max: 999, count: 0 },
      { range: "100-149", min: 100, max: 149, count: 0 },
      { range: "50-99", min: 50, max: 99, count: 0 },
      { range: "1-49", min: 1, max: 49, count: 0 },
      { range: "0", min: 0, max: 0, count: 0 }
    ];
    dashboardData.top.forEach(stock => {
      const range = ranges.find(r => stock.final_score >= r.min && stock.final_score <= r.max);
      if (range) range.count++;
    });
    return ranges;
  }, [dashboardData]);

  // 策略触发率
  const strategyTriggerRate = useMemo(() => {
    if (!dashboardData?.top || dashboardData.top.length === 0) return [];
    const total = dashboardData.top.length;
    return [
      { name: "策略A", value: Math.round((stocksWithSignals.strategyA / total) * 100) },
      { name: "策略B", value: Math.round((stocksWithSignals.strategyB / total) * 100) },
      { name: "策略C", value: Math.round((stocksWithSignals.strategyC / total) * 100) }
    ];
  }, [dashboardData, stocksWithSignals]);

  const toggleStrategy = (strategy: string) => {
    const newSet = new Set(expandedStrategies);
    if (newSet.has(strategy)) {
      newSet.delete(strategy);
    } else {
      newSet.add(strategy);
    }
    setExpandedStrategies(newSet);
  };

  if (!dashboardData) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-700">无法加载仪表盘数据，请确保已运行 a6_models.py 脚本</span>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        {/* 标题 */}
        <div>
          <h1 className="text-3xl font-bold">模型仪表盘</h1>
          <p className="text-gray-600 mt-2">AI 选股策略信号分析系统 | 数据更新时间: {new Date(dashboardData.generated_at).toLocaleString()}</p>
        </div>

        {/* 策略说明卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(STRATEGY_INFO).map(([key, info]) => (
            <Card key={key} className="border-l-4" style={{ borderLeftColor: info.color }}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">{info.name}</CardTitle>
                    <CardDescription className="text-xs mt-1">{info.scoreRange}</CardDescription>
                  </div>
                  <button 
                    onClick={() => toggleStrategy(key)}
                    className="p-1 hover:bg-gray-100 rounded"
                  >
                    {expandedStrategies.has(key) ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>
              </CardHeader>
              {expandedStrategies.has(key) && (
                <CardContent className="space-y-3">
                  <p className="text-sm text-gray-700">{info.description}</p>
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-gray-600">信号类型：</p>
                    {info.signals.map((sig, idx) => (
                      <div key={idx} className="text-xs bg-gray-50 p-2 rounded">
                        <span className="font-semibold">{sig.name}</span>
                        {sig.priority && <Badge variant="outline" className="ml-2 text-xs">{sig.priority}</Badge>}
                        <p className="text-gray-600 mt-1">{sig.desc}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
        </div>

        {/* 信号统计卡片 */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">有信号股票</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stocksWithSignals.total}</div>
              <p className="text-xs text-gray-500 mt-1">触发任一策略</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-blue-600">策略A信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{stocksWithSignals.strategyA}</div>
              <p className="text-xs text-gray-500 mt-1">MA交叉</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-red-600">策略B信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{stocksWithSignals.strategyB}</div>
              <p className="text-xs text-gray-500 mt-1">RSI超卖</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-purple-600">策略C信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">{stocksWithSignals.strategyC}</div>
              <p className="text-xs text-gray-500 mt-1">高级指标</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-orange-600">双策略共振</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-600">{stocksWithSignals.dual}</div>
              <p className="text-xs text-gray-500 mt-1">高价值信号</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-green-600">三策略共振</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stocksWithSignals.triple}</div>
              <p className="text-xs text-gray-500 mt-1">极高价值</p>
            </CardContent>
          </Card>
        </div>

        {/* 搜索和过滤 */}
        <Card>
          <CardHeader>
            <CardTitle>信号列表</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
                <Input
                  placeholder="搜索代码或名称..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={marketFilter} onValueChange={setMarketFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="选择市场" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部市场</SelectItem>
                  <SelectItem value="sh">上海</SelectItem>
                  <SelectItem value="sz">深圳</SelectItem>
                  <SelectItem value="bj">北京</SelectItem>
                </SelectContent>
              </Select>
              <Select value={scoreFilter} onValueChange={setScoreFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="选择信号强度" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部</SelectItem>
                  <SelectItem value="high">强信号 (150+)</SelectItem>
                  <SelectItem value="medium">中等 (50-149)</SelectItem>
                  <SelectItem value="low">弱信号 (1-49)</SelectItem>
                  <SelectItem value="none">无信号</SelectItem>
                </SelectContent>
              </Select>
              <div className="text-sm text-gray-600 flex items-center">
                共 {filteredData.length} 只股票
              </div>
            </div>

            {/* 信号表格 */}
            <div className="border rounded-lg overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="bg-gray-50">
                    <TableHead className="w-20">市场</TableHead>
                    <TableHead className="w-24">代码</TableHead>
                    <TableHead className="w-32">名称</TableHead>
                    <TableHead className="w-24">综合评分</TableHead>
                    <TableHead className="w-20">策略A</TableHead>
                    <TableHead className="w-20">策略B</TableHead>
                    <TableHead className="w-20">策略C</TableHead>
                    <TableHead className="w-32">信号强度</TableHead>
                    <TableHead className="w-24">数据日期</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredData.slice(0, 50).map((stock, idx) => (
                    <TableRow 
                      key={idx}
                      className="hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedStock(stock)}
                    >
                      <TableCell className="font-medium">
                        <Badge variant={stock.market === "sh" ? "default" : stock.market === "sz" ? "secondary" : "outline"}>
                          {stock.market.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-semibold">{stock.code}</TableCell>
                      <TableCell>{stock.name}</TableCell>
                      <TableCell className="font-bold text-lg">
                        <span style={{ color: stock.final_score >= 100 ? "#22c55e" : stock.final_score >= 50 ? "#f59e0b" : "#ef4444" }}>
                          {stock.final_score.toFixed(0)}
                        </span>
                      </TableCell>
                      <TableCell className="text-center">
                        {stock.score_A > 0 ? <Badge variant="outline" className="bg-blue-50">{stock.score_A.toFixed(0)}</Badge> : "-"}
                      </TableCell>
                      <TableCell className="text-center">
                        {stock.score_B > 0 ? <Badge variant="outline" className="bg-red-50">{stock.score_B.toFixed(0)}</Badge> : "-"}
                      </TableCell>
                      <TableCell className="text-center">
                        {(stock.score_C ?? 0) > 0 ? <Badge variant="outline" className="bg-purple-50">{(stock.score_C ?? 0).toFixed(0)}</Badge> : "-"}
                      </TableCell>
                      <TableCell>
                        {stock.final_score >= 100 && <Badge className="bg-green-600">极强</Badge>}
                        {stock.final_score >= 50 && stock.final_score < 100 && <Badge className="bg-orange-600">强</Badge>}
                        {stock.final_score > 0 && stock.final_score < 50 && <Badge variant="outline">弱</Badge>}
                        {stock.final_score === 0 && <Badge variant="outline" className="text-gray-400">无</Badge>}
                      </TableCell>
                      <TableCell className="text-sm text-gray-600">
                        {new Date(stock.last_date).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {filteredData.length > 50 && (
              <div className="text-center text-sm text-gray-600 p-4">
                显示前 50 条记录，共 {filteredData.length} 条
              </div>
            )}
          </CardContent>
        </Card>

        {/* 图表分析 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* 评分分布 */}
          <Card>
            <CardHeader>
              <CardTitle>评分区间分布</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={scoreDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* 策略触发率 */}
          <Card>
            <CardHeader>
              <CardTitle>策略触发率 (%)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={strategyTriggerRate}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="value" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* 选中股票详情 */}
        {selectedStock && (
          <Card>
            <CardHeader>
              <CardTitle>
                {selectedStock.code} - {selectedStock.name}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-600">市场</p>
                  <p className="text-lg font-semibold">{selectedStock.market.toUpperCase()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">综合评分</p>
                  <p className="text-lg font-semibold">{selectedStock.final_score.toFixed(0)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">数据日期</p>
                  <p className="text-lg font-semibold">{new Date(selectedStock.last_date).toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">信号数量</p>
                  <p className="text-lg font-semibold">{selectedStock.signals_count}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                <div className="text-center">
                  <p className="text-sm text-gray-600">策略A评分</p>
                  <p className="text-2xl font-bold text-blue-600">{selectedStock.score_A.toFixed(0)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">策略B评分</p>
                  <p className="text-2xl font-bold text-red-600">{selectedStock.score_B.toFixed(0)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">策略C评分</p>
                  <p className="text-2xl font-bold text-purple-600">{(selectedStock.score_C ?? 0).toFixed(0)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
