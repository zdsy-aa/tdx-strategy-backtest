import { useState, useEffect } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, TrendingUp, Activity, AlertTriangle } from "lucide-react";
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

export default function ModelDashboard() {
  const dashboardData = dashboardDataRaw as unknown as DashboardData;

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
                <CardTitle>评分排行榜</CardTitle>
                <CardDescription>前 20 只股票的综合评分排行</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={scoreDistribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" stroke="rgba(255,255,255,0.5)" />
                    <YAxis dataKey="name" type="category" width={150} stroke="rgba(255,255,255,0.5)" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(0,0,0,0.8)",
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                    <Bar dataKey="评分" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>排行详情</CardTitle>
                <CardDescription>前 50 只股票的详细信息</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>排名</TableCell>
                        <TableCell>代码</TableCell>
                        <TableCell>名称</TableCell>
                        <TableCell>综合评分</TableCell>
                        <TableCell>策略A</TableCell>
                        <TableCell>策略B</TableCell>
                        <TableCell>信号数</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {dashboardData.top.slice(0, 50).map((item, idx) => (
                        <TableRow key={idx}>
                          <TableCell className="font-medium">{idx + 1}</TableCell>
                          <TableCell>{item.code}</TableCell>
                          <TableCell>{item.name}</TableCell>
                          <TableCell>
                            <Badge variant="default">{item.final_score.toFixed(2)}</Badge>
                          </TableCell>
                          <TableCell>{item.score_A.toFixed(2)}</TableCell>
                          <TableCell>{item.score_B.toFixed(2)}</TableCell>
                          <TableCell>{item.signals_count}</TableCell>
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
                <CardTitle>策略A vs 策略B</CardTitle>
                <CardDescription>两个策略的评分对比</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={strategyComparison}>
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
                    <Bar dataKey="策略A" fill="#3b82f6" />
                    <Bar dataKey="策略B" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 市场分布标签 */}
          <TabsContent value="markets" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>市场分布详情</CardTitle>
                <CardDescription>各市场的股票分布情况</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {marketData.map((market, idx) => (
                    <div key={idx} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-semibold">{market.name}</h3>
                        <Badge variant="outline">{market.total} 只股票</Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                          <p className="text-sm text-muted-foreground">成功</p>
                          <p className="text-2xl font-bold text-green-500">{market.成功}</p>
                        </div>
                        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                          <p className="text-sm text-muted-foreground">失败</p>
                          <p className="text-2xl font-bold text-red-500">{market.失败}</p>
                        </div>
                        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                          <p className="text-sm text-muted-foreground">成功率</p>
                          <p className="text-2xl font-bold text-blue-500">
                            {((market.成功 / market.total) * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
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
