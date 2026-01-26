import { useState, useMemo } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, Search, Calendar, TrendingUp, Target, Zap } from "lucide-react";
import Layout from "@/components/Layout";
import forecastSummaryRaw from "@/data/forecast_summary.json";

interface ForecastData {
  code: string;
  name: string;
  latest_close: number;
  kalman_price: number;
  particle_price: number;
  market_state: number;
  ensemble_forecast: number;
  forecast_change_pct: number;
  confidence: number;
  analysis_date: string;
  forecast_date: string;
  // 兼容精简版字段名
  forecast_price?: number;
}

interface SummaryData {
  generated_at: string;
  total_stocks?: number;
  successful?: number;
  failed?: number;
  all_predictions?: ForecastData[];
  // 兼容精简版字段名
  predictions?: ForecastData[];
}

export default function ForecastDashboard() {
  const summaryRaw = forecastSummaryRaw as unknown as SummaryData;
  const summaryData = useMemo(() => {
    return {
      generated_at: summaryRaw.generated_at,
      total_stocks: summaryRaw.total_stocks ?? (summaryRaw.predictions?.length || 0),
      successful: summaryRaw.successful ?? (summaryRaw.predictions?.length || 0),
      failed: summaryRaw.failed ?? 0,
      all_predictions: summaryRaw.all_predictions ?? summaryRaw.predictions ?? []
    };
  }, [summaryRaw]);
  
  // 状态管理
  const [searchTerm, setSearchTerm] = useState("");
  const [dateFilter, setDateFilter] = useState("all");
  const [selectedStock, setSelectedStock] = useState<ForecastData | null>(
    summaryData?.all_predictions?.[0] || null
  );

  // 获取所有可用日期
  const availableDates = useMemo(() => {
    if (!summaryData?.all_predictions) return [];
    const dates = Array.from(new Set(summaryData.all_predictions.map(p => p.analysis_date)));
    return dates.sort((a, b) => b.localeCompare(a));
  }, [summaryData]);

  // 过滤后的数据
  const filteredData = useMemo(() => {
    if (!summaryData?.all_predictions) return [];
    return summaryData.all_predictions.filter(item => {
      const matchesSearch = item.code.includes(searchTerm) || item.name.includes(searchTerm);
      const matchesDate = dateFilter === "all" || item.analysis_date === dateFilter;
      return matchesSearch && matchesDate;
    });
  }, [summaryData, searchTerm, dateFilter]);

  if (!summaryData) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="size-5 text-red-500" />
            <p className="text-red-500">无法加载预测数据。请确保已运行 a7_advanced_forecast.py 脚本。</p>
          </div>
        </div>
      </Layout>
    );
  }

  // 准备图表数据 (基于过滤后的前20条)
  const chartData = filteredData.slice(0, 20);
  
  const forecastComparison = chartData.map((item) => ({
    code: item.code,
    当前价格: item.latest_close,
    次日预测: item.ensemble_forecast,
    变化幅度: item.forecast_change_pct,
  }));

  const confidenceData = chartData.slice(0, 15).map((item) => ({
    code: `${item.code}`,
    置信度: (item.confidence * 100).toFixed(0),
    变化幅度: item.forecast_change_pct,
  }));

  const marketStateDistribution = [
    { state: "牛市", count: filteredData.filter((p) => p.market_state === 0).length },
    { state: "熊市", count: filteredData.filter((p) => p.market_state === 1).length },
    { state: "震荡", count: filteredData.filter((p) => p.market_state === 2).length },
  ];

  const successRate = ((summaryData.successful / summaryData.total_stocks) * 100).toFixed(2);

  const getMarketStateLabel = (state: number) => {
    const labels = ["牛市", "熊市", "震荡"];
    return labels[state] || "未知";
  };

  const getMarketStateColor = (state: number) => {
    const colors = ["text-green-500", "text-red-500", "text-yellow-500"];
    return colors[state] || "text-gray-500";
  };

  return (
    <Layout>
      <div className="space-y-8">
        {/* 页面标题 */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold mb-2">📊 次日价格预测</h1>
            <p className="text-muted-foreground">
              基于多模型集成预测最新数据日期的次日表现 | 最后更新: {new Date(summaryData.generated_at).toLocaleString()}
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input 
                placeholder="搜索代码或名称..." 
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="flex items-center gap-2">
              <Calendar className="size-4 text-muted-foreground" />
              <Select value={dateFilter} onValueChange={setDateFilter}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="选择日期" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部日期</SelectItem>
                  {availableDates.map(date => (
                    <SelectItem key={date} value={date}>{date}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">总股票数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-500">{summaryData.total_stocks.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">全市场覆盖</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/10 to-green-600/5 border-green-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">成功预测</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500">{summaryData.successful.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">成功率 {successRate}%</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">当前筛选</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-500">{filteredData.length.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">符合条件的股票</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">平均置信度</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-500">
                {(
                  (filteredData.slice(0, 100).reduce((sum, p) => sum + p.confidence, 0) /
                    Math.max(1, Math.min(100, filteredData.length))) *
                  100
                ).toFixed(1)}
                %
              </div>
              <p className="text-xs text-muted-foreground mt-1">前100只平均值</p>
            </CardContent>
          </Card>
        </div>

        {/* 标签页面 */}
        <Tabs defaultValue="predictions" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="predictions">预测排行</TabsTrigger>
            <TabsTrigger value="analysis">次日预测分析</TabsTrigger>
            <TabsTrigger value="overview">市场概览</TabsTrigger>
            <TabsTrigger value="details">详细信息</TabsTrigger>
          </TabsList>

          {/* 预测排行标签 */}
          <TabsContent value="predictions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>预测排行榜</CardTitle>
                <CardDescription>按预测涨幅排序的股票列表 (当前筛选: {filteredData.length} 只)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-16">排名</TableHead>
                        <TableHead className="w-24">代码</TableHead>
                        <TableHead className="w-32">名称</TableHead>
                        <TableHead className="text-right">当前价格</TableHead>
                        <TableHead className="text-right">次日预测</TableHead>
                        <TableHead className="text-center">预测涨幅</TableHead>
                        <TableHead className="text-center">置信度</TableHead>
                        <TableHead className="text-center">市场状态</TableHead>
                        <TableHead className="text-right">分析日期</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredData.slice(0, 100).map((item, idx) => (
                        <TableRow 
                          key={idx} 
                          className={`cursor-pointer hover:bg-white/5 transition-all ${
                            selectedStock?.code === item.code 
                              ? "ring-2 ring-primary ring-inset z-10" 
                              : ""
                          }`} 
                          onClick={() => setSelectedStock(item)}
                        >
                          <TableCell className="font-medium">{idx + 1}</TableCell>
                          <TableCell>{item.code}</TableCell>
                          <TableCell>{item.name}</TableCell>
                          <TableCell className="text-right font-mono">¥{item.latest_close.toFixed(2)}</TableCell>
                          <TableCell className="text-right font-mono text-primary">¥{(item.ensemble_forecast ?? item.forecast_price ?? 0).toFixed(2)}</TableCell>
                          <TableCell className="text-center">
                            <Badge variant={item.forecast_change_pct > 0 ? "default" : "secondary"} className={item.forecast_change_pct > 0 ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600"}>
                              {item.forecast_change_pct > 0 ? "+" : ""}
                              {item.forecast_change_pct.toFixed(2)}%
                            </Badge>
                          </TableCell>
                          <TableCell className="text-center">
                            <Badge variant="outline">{(item.confidence * 100).toFixed(0)}%</Badge>
                          </TableCell>
                          <TableCell className={`text-center font-medium ${getMarketStateColor(item.market_state)}`}>
                            {getMarketStateLabel(item.market_state)}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground text-xs">{item.analysis_date}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {filteredData.length > 100 && (
                    <p className="text-center text-sm text-muted-foreground mt-4">仅显示前 100 条结果，请使用搜索或日期过滤缩小范围</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 预测分析标签 */}
          <TabsContent value="analysis" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>次日预测价格对比</CardTitle>
                <CardDescription>当前筛选前 20 只股票的价格对比</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={forecastComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="code" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="当前价格" fill="#3b82f6" />
                    <Bar dataKey="次日预测" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>次日预测涨幅分布</CardTitle>
                <CardDescription>预测变化百分比</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={forecastComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="code" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Bar dataKey="变化幅度" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 概览标签 */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>市场状态分布</CardTitle>
                  <CardDescription>当前筛选范围内的市场状态统计</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={marketStateDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="state" stroke="rgba(255,255,255,0.5)" />
                      <YAxis stroke="rgba(255,255,255,0.5)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(0,0,0,0.8)",
                          border: "1px solid rgba(255,255,255,0.2)",
                        }}
                      />
                      <Bar dataKey="count" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>预测置信度对比</CardTitle>
                  <CardDescription>前 15 只股票的置信度</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={confidenceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="code" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
                      <YAxis stroke="rgba(255,255,255,0.5)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(0,0,0,0.8)",
                          border: "1px solid rgba(255,255,255,0.2)",
                        }}
                      />
                      <Bar dataKey="置信度" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 详细信息标签 */}
          <TabsContent value="details" className="space-y-6">
            {selectedStock ? (
              <Card>
                <CardHeader>
                  <CardTitle>
                    {selectedStock.code} - {selectedStock.name}
                  </CardTitle>
                  <CardDescription>分析日期: {selectedStock.analysis_date} | 预测目标日期: {selectedStock.forecast_date}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <p className="text-sm text-muted-foreground">当前收盘价</p>
                      <p className="text-2xl font-bold text-blue-500">¥{selectedStock.latest_close.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                      <p className="text-sm text-muted-foreground">次日预测价</p>
                      <p className="text-2xl font-bold text-green-500">¥{selectedStock.ensemble_forecast.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <p className="text-sm text-muted-foreground">预测涨跌幅</p>
                      <p className={`text-2xl font-bold ${selectedStock.forecast_change_pct > 0 ? "text-green-500" : "text-red-500"}`}>
                        {selectedStock.forecast_change_pct > 0 ? "+" : ""}{selectedStock.forecast_change_pct.toFixed(2)}%
                      </p>
                    </div>

                    <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                      <p className="text-sm text-muted-foreground">卡尔曼平滑价</p>
                      <p className="text-xl font-semibold">¥{selectedStock.kalman_price.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                      <p className="text-sm text-muted-foreground">粒子滤波预测</p>
                      <p className="text-xl font-semibold">¥{selectedStock.particle_price.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                      <p className="text-sm text-muted-foreground">市场状态 (HMM)</p>
                      <p className={`text-xl font-semibold ${getMarketStateColor(selectedStock.market_state)}`}>
                        {getMarketStateLabel(selectedStock.market_state)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-4 rounded-lg bg-primary/5 border border-primary/10">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="size-5 text-primary" />
                      <h4 className="font-semibold">预测置信度分析</h4>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      该预测基于随机森林集成模型，结合了卡尔曼滤波平滑、粒子滤波趋势以及隐马尔可夫市场状态识别。
                      当前置信度为 <span className="font-bold text-primary">{(selectedStock.confidence * 100).toFixed(0)}%</span>。
                    </p>
                    <div className="w-full bg-white/10 rounded-full h-2">
                      <div 
                        className="bg-primary h-2 rounded-full transition-all" 
                        style={{ width: `${selectedStock.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                请在排行列表中选择一只股票查看详细预测分析
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
