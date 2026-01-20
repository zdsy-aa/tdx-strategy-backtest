import { useState, useMemo } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, TrendingUp, Activity, Search, Info, Zap, Target, BarChart3 } from "lucide-react";
import Layout from "@/components/Layout";
import dashboardDataRaw from "@/data/dashboard.json";

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
  }
};

export default function ModelDashboard() {
  const dashboardData = dashboardDataRaw as unknown as DashboardData;
  const [searchTerm, setSearchTerm] = useState("");
  const [marketFilter, setMarketFilter] = useState("all");
  const [scoreFilter, setScoreFilter] = useState("all");
  const [selectedStock, setSelectedStock] = useState<typeof dashboardData.top[0] | null>(null);

  // 过滤数据
  const filteredData = useMemo(() => {
    if (!dashboardData?.top) return [];
    return dashboardData.top.filter(item => {
      const matchesSearch = item.code.includes(searchTerm) || item.name.includes(searchTerm);
      const matchesMarket = marketFilter === "all" || item.market === marketFilter;
      const matchesScore = scoreFilter === "all" || 
        (scoreFilter === "high" && item.final_score >= 60) ||
        (scoreFilter === "medium" && item.final_score >= 30 && item.final_score < 60) ||
        (scoreFilter === "low" && item.final_score < 30 && item.final_score > 0) ||
        (scoreFilter === "none" && item.final_score === 0);
      return matchesSearch && matchesMarket && matchesScore;
    });
  }, [dashboardData, searchTerm, marketFilter, scoreFilter]);

  // 统计有信号的股票
  const stocksWithSignals = useMemo(() => {
    if (!dashboardData?.top) return { total: 0, strategyA: 0, strategyB: 0, both: 0 };
    const withA = dashboardData.top.filter(s => s.score_A > 0).length;
    const withB = dashboardData.top.filter(s => s.score_B > 0).length;
    const withBoth = dashboardData.top.filter(s => s.score_A > 0 && s.score_B > 0).length;
    const withAny = dashboardData.top.filter(s => s.final_score > 0).length;
    return { total: withAny, strategyA: withA, strategyB: withB, both: withBoth };
  }, [dashboardData]);

  if (!dashboardData) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="size-5 text-red-500" />
            <p className="text-red-500">无法加载模型仪表盘数据。请确保已运行 a6_models.py 脚本。</p>
          </div>
        </div>
      </Layout>
    );
  }

  // 准备图表数据
  const marketData = Object.entries(dashboardData.markets).map(([market, stats]) => ({
    name: market.toUpperCase(),
    成功: stats.ok,
    失败: stats.fail,
    total: stats.total,
  }));

  // 有信号股票的评分分布
  const signalStocks = filteredData.filter(s => s.final_score > 0);
  const scoreDistribution = signalStocks.slice(0, 20).map((item) => ({
    name: `${item.code}`,
    fullName: `${item.code}-${item.name}`,
    综合评分: parseFloat(item.final_score.toFixed(1)),
    MA交叉: parseFloat(item.score_A.toFixed(1)),
    RSI超卖: parseFloat(item.score_B.toFixed(1)),
  }));

  const strategyComparison = signalStocks.slice(0, 15).map((item) => ({
    name: `${item.code}`,
    [STRATEGY_INFO.A.shortName]: parseFloat(item.score_A.toFixed(1)),
    [STRATEGY_INFO.B.shortName]: parseFloat(item.score_B.toFixed(1)),
  }));

  // 信号类型分布
  const signalTypeData = [
    { name: "仅MA交叉", value: stocksWithSignals.strategyA - stocksWithSignals.both, color: STRATEGY_INFO.A.color },
    { name: "仅RSI超卖", value: stocksWithSignals.strategyB - stocksWithSignals.both, color: STRATEGY_INFO.B.color },
    { name: "双策略共振", value: stocksWithSignals.both, color: "#10b981" },
  ];

  const successRate = (
    (dashboardData.counts.symbols_ok / dashboardData.counts.symbols_total) *
    100
  ).toFixed(2);

  const getScoreBadge = (score: number) => {
    if (score >= 60) return <Badge className="bg-green-500 hover:bg-green-600">强信号</Badge>;
    if (score >= 30) return <Badge className="bg-yellow-500 hover:bg-yellow-600">中等</Badge>;
    if (score > 0) return <Badge className="bg-blue-500 hover:bg-blue-600">弱信号</Badge>;
    return <Badge variant="secondary">无信号</Badge>;
  };

  const getMarketName = (market: string) => {
    const names: Record<string, string> = { sh: "上海", sz: "深圳", bj: "北京" };
    return names[market] || market.toUpperCase();
  };

  return (
    <Layout>
      <div className="space-y-8">
        {/* 页面标题 */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold mb-2">📈 策略信号仪表盘</h1>
            <p className="text-muted-foreground">
              基于技术指标的多策略信号监控系统 | 最后更新: {new Date(dashboardData.generated_at).toLocaleString()}
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative w-48">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input 
                placeholder="搜索代码或名称..." 
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <Select value={marketFilter} onValueChange={setMarketFilter}>
              <SelectTrigger className="w-28">
                <SelectValue placeholder="市场" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部市场</SelectItem>
                <SelectItem value="sh">上海</SelectItem>
                <SelectItem value="sz">深圳</SelectItem>
                <SelectItem value="bj">北京</SelectItem>
              </SelectContent>
            </Select>

            <Select value={scoreFilter} onValueChange={setScoreFilter}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="信号强度" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部</SelectItem>
                <SelectItem value="high">强信号(≥60)</SelectItem>
                <SelectItem value="medium">中等(30-60)</SelectItem>
                <SelectItem value="low">弱信号(1-30)</SelectItem>
                <SelectItem value="none">无信号</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* 策略说明卡片 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="size-5 text-blue-500" />
                <CardTitle className="text-lg">{STRATEGY_INFO.A.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-3">{STRATEGY_INFO.A.description}</p>
              <div className="space-y-2">
                {STRATEGY_INFO.A.signals.map((sig, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <Badge variant="outline" className="shrink-0">{sig.name}</Badge>
                    <span className="text-muted-foreground">{sig.desc}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/10 flex justify-between text-sm">
                <span className="text-muted-foreground">评分范围</span>
                <span className="font-medium text-blue-500">{STRATEGY_INFO.A.scoreRange}</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Activity className="size-5 text-red-500" />
                <CardTitle className="text-lg">{STRATEGY_INFO.B.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-3">{STRATEGY_INFO.B.description}</p>
              <div className="space-y-2">
                {STRATEGY_INFO.B.signals.map((sig, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <Badge variant="outline" className="shrink-0">{sig.name}</Badge>
                    <span className="text-muted-foreground">{sig.desc}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/10 flex justify-between text-sm">
                <span className="text-muted-foreground">评分范围</span>
                <span className="font-medium text-red-500">{STRATEGY_INFO.B.scoreRange}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 信号统计卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">有信号股票</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-500">{stocksWithSignals.total}</div>
              <p className="text-xs text-muted-foreground mt-1">
                占比 {((stocksWithSignals.total / dashboardData.counts.symbols_ok) * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">MA交叉信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-500">{stocksWithSignals.strategyA}</div>
              <p className="text-xs text-muted-foreground mt-1">趋势跟踪策略触发</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">RSI超卖信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-500">{stocksWithSignals.strategyB}</div>
              <p className="text-xs text-muted-foreground mt-1">超卖反弹策略触发</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/10 to-green-600/5 border-green-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">双策略共振</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500">{stocksWithSignals.both}</div>
              <p className="text-xs text-muted-foreground mt-1">两策略同时触发</p>
            </CardContent>
          </Card>
        </div>

        {/* 标签页面 */}
        <Tabs defaultValue="signals" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="signals">信号列表</TabsTrigger>
            <TabsTrigger value="analysis">策略分析</TabsTrigger>
            <TabsTrigger value="distribution">信号分布</TabsTrigger>
            <TabsTrigger value="markets">市场统计</TabsTrigger>
          </TabsList>

          {/* 信号列表标签 */}
          <TabsContent value="signals" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>策略信号排行榜</CardTitle>
                <CardDescription>
                  按综合评分排序的股票列表 (当前筛选: {filteredData.length} 只，有信号: {filteredData.filter(s => s.final_score > 0).length} 只)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-16">排名</TableHead>
                        <TableHead className="w-20">市场</TableHead>
                        <TableHead className="w-24">代码</TableHead>
                        <TableHead className="w-32">名称</TableHead>
                        <TableHead className="text-center">信号强度</TableHead>
                        <TableHead className="text-right">综合评分</TableHead>
                        <TableHead className="text-right">MA交叉</TableHead>
                        <TableHead className="text-right">RSI超卖</TableHead>
                        <TableHead className="text-center">信号数</TableHead>
                        <TableHead className="text-right">数据日期</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredData.slice(0, 100).map((item, idx) => (
                        <TableRow 
                          key={idx} 
                          className={`cursor-pointer hover:bg-white/5 ${item.final_score === 0 ? 'opacity-50' : ''}`}
                          onClick={() => setSelectedStock(item)}
                        >
                          <TableCell className="font-medium">{idx + 1}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{getMarketName(item.market)}</Badge>
                          </TableCell>
                          <TableCell className="font-mono">{item.code}</TableCell>
                          <TableCell>{item.name}</TableCell>
                          <TableCell className="text-center">{getScoreBadge(item.final_score)}</TableCell>
                          <TableCell className="text-right font-bold">{item.final_score.toFixed(1)}</TableCell>
                          <TableCell className="text-right">
                            <span className={item.score_A > 0 ? "text-blue-500 font-medium" : "text-muted-foreground"}>
                              {item.score_A.toFixed(1)}
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            <span className={item.score_B > 0 ? "text-red-500 font-medium" : "text-muted-foreground"}>
                              {item.score_B.toFixed(1)}
                            </span>
                          </TableCell>
                          <TableCell className="text-center">
                            <Badge variant="secondary">{item.signals_count}</Badge>
                          </TableCell>
                          <TableCell className="text-right text-xs text-muted-foreground">
                            {new Date(item.last_date).toLocaleDateString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {filteredData.length > 100 && (
                    <p className="text-center text-sm text-muted-foreground mt-4">
                      仅显示前 100 条结果，请使用搜索或筛选缩小范围
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* 选中股票详情 */}
            {selectedStock && selectedStock.final_score > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>{selectedStock.code} - {selectedStock.name}</CardTitle>
                  <CardDescription>
                    {getMarketName(selectedStock.market)}市场 | 数据日期: {new Date(selectedStock.last_date).toLocaleDateString()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <p className="text-sm text-muted-foreground">综合评分</p>
                      <p className="text-3xl font-bold text-purple-500">{selectedStock.final_score.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground mt-1">两策略评分之和</p>
                    </div>
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <p className="text-sm text-muted-foreground">{STRATEGY_INFO.A.shortName}评分</p>
                      <p className="text-3xl font-bold text-blue-500">{selectedStock.score_A.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {selectedStock.score_A > 0 ? "MA5上穿MA20触发" : "未触发"}
                      </p>
                    </div>
                    <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                      <p className="text-sm text-muted-foreground">{STRATEGY_INFO.B.shortName}评分</p>
                      <p className="text-3xl font-bold text-red-500">{selectedStock.score_B.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {selectedStock.score_B >= 40 ? "RSI超卖触发" : selectedStock.score_B > 0 ? "放量信号触发" : "未触发"}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* 策略分析标签 */}
          <TabsContent value="analysis" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>有信号股票评分对比</CardTitle>
                  <CardDescription>前20只有信号股票的两策略评分</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={scoreDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={60} />
                      <YAxis stroke="rgba(255,255,255,0.5)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(0,0,0,0.8)",
                          border: "1px solid rgba(255,255,255,0.2)",
                        }}
                        formatter={(value: number, name: string) => [value.toFixed(1), name]}
                      />
                      <Legend />
                      <Bar dataKey="MA交叉" fill={STRATEGY_INFO.A.color} />
                      <Bar dataKey="RSI超卖" fill={STRATEGY_INFO.B.color} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>信号类型分布</CardTitle>
                  <CardDescription>各策略触发情况统计</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={signalTypeData.filter(d => d.value > 0)}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {signalTypeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="mt-4 space-y-2">
                    {signalTypeData.map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                          <span>{item.name}</span>
                        </div>
                        <span className="font-medium">{item.value} 只</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>综合评分排行</CardTitle>
                <CardDescription>有信号股票的综合评分分布</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={scoreDistribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
                    <YAxis dataKey="name" type="category" width={80} stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Bar dataKey="综合评分" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 信号分布标签 */}
          <TabsContent value="distribution" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>评分区间分布</CardTitle>
                  <CardDescription>按信号强度分类的股票数量</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      { label: "强信号 (≥60分)", count: dashboardData.top.filter(s => s.final_score >= 60).length, color: "bg-green-500" },
                      { label: "中等信号 (30-60分)", count: dashboardData.top.filter(s => s.final_score >= 30 && s.final_score < 60).length, color: "bg-yellow-500" },
                      { label: "弱信号 (1-30分)", count: dashboardData.top.filter(s => s.final_score > 0 && s.final_score < 30).length, color: "bg-blue-500" },
                      { label: "无信号 (0分)", count: dashboardData.top.filter(s => s.final_score === 0).length, color: "bg-gray-500" },
                    ].map((item, idx) => (
                      <div key={idx} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>{item.label}</span>
                          <span className="font-medium">{item.count} 只</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-2">
                          <div 
                            className={`${item.color} h-2 rounded-full transition-all`}
                            style={{ width: `${(item.count / dashboardData.top.length) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>策略触发统计</CardTitle>
                  <CardDescription>各策略的触发率</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="size-4 text-blue-500" />
                          <span>{STRATEGY_INFO.A.name}</span>
                        </div>
                        <span className="font-medium text-blue-500">{stocksWithSignals.strategyA} 只</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-3">
                        <div 
                          className="bg-blue-500 h-3 rounded-full transition-all"
                          style={{ width: `${(stocksWithSignals.strategyA / dashboardData.counts.symbols_ok) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        触发率: {((stocksWithSignals.strategyA / dashboardData.counts.symbols_ok) * 100).toFixed(2)}%
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Activity className="size-4 text-red-500" />
                          <span>{STRATEGY_INFO.B.name}</span>
                        </div>
                        <span className="font-medium text-red-500">{stocksWithSignals.strategyB} 只</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-3">
                        <div 
                          className="bg-red-500 h-3 rounded-full transition-all"
                          style={{ width: `${(stocksWithSignals.strategyB / dashboardData.counts.symbols_ok) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        触发率: {((stocksWithSignals.strategyB / dashboardData.counts.symbols_ok) * 100).toFixed(2)}%
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Zap className="size-4 text-green-500" />
                          <span>双策略共振</span>
                        </div>
                        <span className="font-medium text-green-500">{stocksWithSignals.both} 只</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-3">
                        <div 
                          className="bg-green-500 h-3 rounded-full transition-all"
                          style={{ width: `${(stocksWithSignals.both / dashboardData.counts.symbols_ok) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        共振率: {((stocksWithSignals.both / dashboardData.counts.symbols_ok) * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 市场统计标签 */}
          <TabsContent value="markets" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>市场处理统计</CardTitle>
                <CardDescription>各市场的数据处理情况</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={marketData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="成功" fill="#10b981" />
                    <Bar dataKey="失败" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {marketData.map((market, idx) => (
                <Card key={idx}>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">{market.name} 市场</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">总股票数</span>
                        <span className="font-bold">{market.total}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">成功处理</span>
                        <span className="font-medium text-green-500">{market.成功}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">处理失败</span>
                        <span className="font-medium text-red-500">{market.失败}</span>
                      </div>
                      <div className="pt-2 border-t border-white/10">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">成功率</span>
                          <span className="font-bold text-primary">
                            {((market.成功 / market.total) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
