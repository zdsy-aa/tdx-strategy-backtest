import { useState, useEffect } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, TrendingUp, Target, Zap } from "lucide-react";
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
}

interface SummaryData {
  generated_at: string;
  total_stocks: number;
  successful: number;
  failed: number;
  top_predictions: ForecastData[];
}

export default function ForecastDashboard() {
  const summaryData = forecastSummaryRaw as unknown as SummaryData;
  const [selectedStock, setSelectedStock] = useState<ForecastData | null>(
    summaryData?.top_predictions?.[0] || null
  );

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

  // 准备图表数据
  const forecastComparison = summaryData.top_predictions.slice(0, 20).map((item) => ({
    code: item.code,
    当前价格: item.latest_close,
    预测价格: item.ensemble_forecast,
    变化幅度: item.forecast_change_pct,
  }));

  const confidenceData = summaryData.top_predictions.slice(0, 15).map((item) => ({
    code: `${item.code}-${item.name}`,
    置信度: (item.confidence * 100).toFixed(0),
    变化幅度: item.forecast_change_pct,
  }));

  const marketStateDistribution = [
    { state: "牛市", count: summaryData.top_predictions.filter((p) => p.market_state === 0).length },
    { state: "熊市", count: summaryData.top_predictions.filter((p) => p.market_state === 1).length },
    { state: "震荡", count: summaryData.top_predictions.filter((p) => p.market_state === 2).length },
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
        <div>
          <h1 className="text-4xl font-bold mb-2">📊 高级预测分析</h1>
          <p className="text-muted-foreground">
            基于卡尔曼滤波、粒子滤波、HMM 和随机森林的多模型集成预测 | 最后更新: {new Date(summaryData.generated_at).toLocaleString()}
          </p>
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
              <CardTitle className="text-sm font-medium text-muted-foreground">预测失败</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-500">{summaryData.failed}</div>
              <p className="text-xs text-muted-foreground mt-1">需要检查</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">平均置信度</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-500">
                {(
                  (summaryData.top_predictions.reduce((sum, p) => sum + p.confidence, 0) /
                    summaryData.top_predictions.length) *
                  100
                ).toFixed(1)}
                %
              </div>
              <p className="text-xs text-muted-foreground mt-1">预测可靠性</p>
            </CardContent>
          </Card>
        </div>

        {/* 标签页面 */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="predictions">预测排行</TabsTrigger>
            <TabsTrigger value="analysis">预测分析</TabsTrigger>
            <TabsTrigger value="details">详细信息</TabsTrigger>
          </TabsList>

          {/* 概览标签 */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>市场状态分布</CardTitle>
                  <CardDescription>当前市场各状态的股票数量</CardDescription>
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
                  <CardTitle>预测置信度分布</CardTitle>
                  <CardDescription>前 15 只股票的置信度对比</CardDescription>
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

            <Card>
              <CardHeader>
                <CardTitle>当前价格 vs 预测价格</CardTitle>
                <CardDescription>前 20 只股票的价格对比</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="当前价格" stroke="rgba(255,255,255,0.5)" />
                    <YAxis dataKey="预测价格" stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Scatter name="预测" data={forecastComparison} fill="#3b82f6" />
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 预测排行标签 */}
          <TabsContent value="predictions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>预测排行榜</CardTitle>
                <CardDescription>按预测涨幅排序的前 50 只股票</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>排名</TableCell>
                        <TableCell>代码</TableCell>
                        <TableCell>名称</TableCell>
                        <TableCell>当前价格</TableCell>
                        <TableCell>预测价格</TableCell>
                        <TableCell>预测涨幅</TableCell>
                        <TableCell>置信度</TableCell>
                        <TableCell>市场状态</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {summaryData.top_predictions.slice(0, 50).map((item, idx) => (
                        <TableRow key={idx} className="cursor-pointer hover:bg-white/5" onClick={() => setSelectedStock(item)}>
                          <TableCell className="font-medium">{idx + 1}</TableCell>
                          <TableCell>{item.code}</TableCell>
                          <TableCell>{item.name}</TableCell>
                          <TableCell>¥{item.latest_close.toFixed(2)}</TableCell>
                          <TableCell>¥{item.ensemble_forecast.toFixed(2)}</TableCell>
                          <TableCell>
                            <Badge variant={item.forecast_change_pct > 0 ? "default" : "secondary"}>
                              {item.forecast_change_pct > 0 ? "+" : ""}
                              {item.forecast_change_pct.toFixed(2)}%
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">{(item.confidence * 100).toFixed(0)}%</Badge>
                          </TableCell>
                          <TableCell className={getMarketStateColor(item.market_state)}>
                            {getMarketStateLabel(item.market_state)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 预测分析标签 */}
          <TabsContent value="analysis" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>预测价格对比</CardTitle>
                <CardDescription>前 20 只股票的当前价格与预测价格对比</CardDescription>
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
                    <Bar dataKey="预测价格" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>预测变化幅度分析</CardTitle>
                <CardDescription>预测涨幅分布情况</CardDescription>
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

          {/* 详细信息标签 */}
          <TabsContent value="details" className="space-y-6">
            {selectedStock && (
              <Card>
                <CardHeader>
                  <CardTitle>
                    {selectedStock.code} - {selectedStock.name}
                  </CardTitle>
                  <CardDescription>分析日期: {selectedStock.analysis_date}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <p className="text-sm text-muted-foreground">当前收盘价</p>
                      <p className="text-2xl font-bold text-blue-500">¥{selectedStock.latest_close.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <p className="text-sm text-muted-foreground">卡尔曼平滑价</p>
                      <p className="text-2xl font-bold text-purple-500">¥{selectedStock.kalman_price.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                      <p className="text-sm text-muted-foreground">粒子滤波预测</p>
                      <p className="text-2xl font-bold text-cyan-500">¥{selectedStock.particle_price.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                      <p className="text-sm text-muted-foreground">集成模型预测</p>
                      <p className="text-2xl font-bold text-green-500">¥{selectedStock.ensemble_forecast.toFixed(2)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/20">
                      <p className="text-sm text-muted-foreground">预测涨幅</p>
                      <p className={`text-2xl font-bold ${selectedStock.forecast_change_pct > 0 ? "text-green-500" : "text-red-500"}`}>
                        {selectedStock.forecast_change_pct > 0 ? "+" : ""}
                        {selectedStock.forecast_change_pct.toFixed(2)}%
                      </p>
                    </div>

                    <div className="p-4 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                      <p className="text-sm text-muted-foreground">预测置信度</p>
                      <p className="text-2xl font-bold text-indigo-500">{(selectedStock.confidence * 100).toFixed(0)}%</p>
                    </div>

                    <div className="p-4 rounded-lg bg-pink-500/10 border border-pink-500/20 md:col-span-2 lg:col-span-1">
                      <p className="text-sm text-muted-foreground">市场状态</p>
                      <p className={`text-2xl font-bold ${getMarketStateColor(selectedStock.market_state)}`}>
                        {getMarketStateLabel(selectedStock.market_state)}
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 p-4 rounded-lg bg-white/5 border border-white/10">
                    <h3 className="font-semibold mb-3">预测模型说明</h3>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li>
                        <strong>卡尔曼滤波：</strong>
                        通过递归算法平滑价格曲线，减少市场噪声，提供更清晰的价格趋势
                      </li>
                      <li>
                        <strong>粒子滤波：</strong>
                        处理非高斯分布的市场数据，通过粒子群模拟价格运动，预测下一时刻价格
                      </li>
                      <li>
                        <strong>隐马尔可夫模型 (HMM)：</strong>
                        识别市场的隐藏状态（牛市、熊市、震荡），捕捉市场的周期性特征
                      </li>
                      <li>
                        <strong>随机森林集成：</strong>
                        结合多种技术指标和滤波结果，通过集成学习进行最终价格预测
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle>其他预测结果</CardTitle>
                <CardDescription>点击选择查看详细信息</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {summaryData.top_predictions.slice(0, 30).map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        selectedStock?.code === item.code
                          ? "bg-blue-500/20 border-blue-500"
                          : "bg-white/5 border-white/10 hover:bg-white/10"
                      }`}
                      onClick={() => setSelectedStock(item)}
                    >
                      <p className="font-semibold">{item.code}</p>
                      <p className="text-sm text-muted-foreground">{item.name}</p>
                      <p className={`text-sm font-medium ${item.forecast_change_pct > 0 ? "text-green-500" : "text-red-500"}`}>
                        {item.forecast_change_pct > 0 ? "+" : ""}
                        {item.forecast_change_pct.toFixed(2)}%
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
