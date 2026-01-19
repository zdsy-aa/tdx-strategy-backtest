import { useState, useEffect } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, TrendingUp, Activity, AlertTriangle } from "lucide-react";
import Layout from "@/components/Layout";

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

export default function ModelDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const response = await import("@/data/dashboard.json");
        setDashboardData(response.default);
        setError(null);
      } catch (err) {
        setError("无法加载模型仪表盘数据。请确保已运行 a6_models.py 脚本。");
        console.error("Error loading dashboard:", err);
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">加载模型数据中...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (error || !dashboardData) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="size-5 text-red-500" />
            <p className="text-red-500">{error || "无法加载数据"}</p>
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

  const scoreDistribution = dashboardData.top.slice(0, 20).map((item) => ({
    name: `${item.code}-${item.name}`,
    评分: parseFloat(item.final_score.toFixed(1)),
  }));

  const strategyComparison = dashboardData.top.slice(0, 15).map((item) => ({
    name: `${item.code}`,
    策略A: parseFloat(item.score_A.toFixed(1)),
    策略B: parseFloat(item.score_B.toFixed(1)),
  }));

  const successRate = (
    (dashboardData.counts.symbols_ok / dashboardData.counts.symbols_total) *
    100
  ).toFixed(2);

  const COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"];

  return (
    <Layout>
      <div className="space-y-8">
        {/* 页面标题 */}
        <div>
          <h1 className="text-4xl font-bold mb-2">AI 模型仪表盘</h1>
          <p className="text-muted-foreground">
            基于 a6_models.py 的策略评分系统 | 最后更新: {new Date(dashboardData.generated_at).toLocaleString()}
          </p>
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">总股票数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-500">{dashboardData.counts.symbols_total.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">全市场覆盖</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/10 to-green-600/5 border-green-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">成功处理</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500">{dashboardData.counts.symbols_ok.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">成功率 {successRate}%</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">处理失败</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-500">{dashboardData.counts.symbols_fail}</div>
              <p className="text-xs text-muted-foreground mt-1">需要检查</p>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">市场分布</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-500">{Object.keys(dashboardData.markets).length}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {Object.keys(dashboardData.markets).map((m) => m.toUpperCase()).join(" / ")}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* 标签页面 */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="ranking">评分排行</TabsTrigger>
            <TabsTrigger value="analysis">策略对比</TabsTrigger>
            <TabsTrigger value="markets">市场分布</TabsTrigger>
          </TabsList>

          {/* 概览标签 */}
          <TabsContent value="overview" className="space-y-6">
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

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>处理成功率</CardTitle>
                  <CardDescription>全市场数据处理成功率</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: "成功", value: dashboardData.counts.symbols_ok },
                          { name: "失败", value: dashboardData.counts.symbols_fail },
                        ]}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        <Cell fill="#10b981" />
                        <Cell fill="#ef4444" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="mt-4 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">成功率</span>
                      <span className="font-semibold text-green-500">{successRate}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">失败率</span>
                      <span className="font-semibold text-red-500">{(100 - parseFloat(successRate)).toFixed(2)}%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>市场分布统计</CardTitle>
                  <CardDescription>各市场的股票数量</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {marketData.map((market, idx) => (
                    <div key={idx} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{market.name}</span>
                        <Badge variant="secondary">{market.total} 只</Badge>
                      </div>
                      <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-full"
                          style={{
                            width: `${(market.成功 / dashboardData.counts.symbols_total) * 100}%`,
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>成功: {market.成功}</span>
                        <span>失败: {market.失败}</span>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 评分排行标签 */}
          <TabsContent value="ranking" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>AI 评分排行 TOP 50</CardTitle>
                <CardDescription>按综合评分从高到低排序</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg border border-white/10 overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 hover:bg-white/5">
                        <TableHead className="w-12">排名</TableHead>
                        <TableHead>市场</TableHead>
                        <TableHead>代码</TableHead>
                        <TableHead>名称</TableHead>
                        <TableHead className="text-right">综合评分</TableHead>
                        <TableHead className="text-right">策略A</TableHead>
                        <TableHead className="text-right">策略B</TableHead>
                        <TableHead className="text-right">信号数</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dashboardData.top.map((stock, idx) => (
                        <TableRow key={`${stock.market}-${stock.code}`} className="border-white/10 hover:bg-white/5">
                          <TableCell className="font-semibold text-primary">
                            {idx + 1}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="uppercase">
                              {stock.market}
                            </Badge>
                          </TableCell>
                          <TableCell className="font-mono font-semibold">{stock.code}</TableCell>
                          <TableCell>{stock.name}</TableCell>
                          <TableCell className="text-right">
                            <span className="inline-flex items-center gap-2">
                              <TrendingUp className="size-4 text-green-500" />
                              <span className="font-semibold text-green-500">
                                {stock.final_score.toFixed(1)}
                              </span>
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            <span className="text-blue-400">{stock.score_A.toFixed(1)}</span>
                          </TableCell>
                          <TableCell className="text-right">
                            <span className="text-purple-400">{stock.score_B.toFixed(1)}</span>
                          </TableCell>
                          <TableCell className="text-right">
                            <Badge variant="secondary">{stock.signals_count}</Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 策略对比标签 */}
          <TabsContent value="analysis" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>策略 A vs 策略 B 对比</CardTitle>
                <CardDescription>前15只股票的两个策略评分对比</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={strategyComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="策略A" fill="#3b82f6" />
                    <Bar dataKey="策略B" fill="#a855f7" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>评分分布</CardTitle>
                <CardDescription>TOP 20 股票的综合评分分布</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scoreDistribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
                    <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Line type="monotone" dataKey="评分" stroke="#10b981" strokeWidth={2} dot={{ fill: "#10b981" }} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 市场分布标签 */}
          <TabsContent value="markets" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {marketData.map((market, idx) => (
                <Card key={idx}>
                  <CardHeader>
                    <CardTitle className="text-lg">{market.name} 市场</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">总数</span>
                        <span className="font-semibold">{market.total}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-green-500">成功</span>
                        <span className="font-semibold text-green-500">{market.成功}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-red-500">失败</span>
                        <span className="font-semibold text-red-500">{market.失败}</span>
                      </div>
                    </div>
                    <div className="pt-4 border-t border-white/10">
                      <div className="text-sm text-muted-foreground mb-2">成功率</div>
                      <div className="text-2xl font-bold text-green-500">
                        {((market.成功 / market.total) * 100).toFixed(1)}%
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
