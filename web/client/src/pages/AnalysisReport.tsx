import { useState, useEffect } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  BarChart3, 
  TrendingUp, 
  Target, 
  PieChart, 
  Activity,
  CheckCircle2,
  XCircle,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Layers,
  LineChart
} from "lucide-react";

// 导入分析数据
import signalSummaryData from "@/data/signal_summary.json";
import patternSummaryData from "@/data/pattern_summary.json";
import patternBySignalData from "@/data/pattern_analysis_by_signal.json";

// 类型定义
interface SignalTypeStats {
  total: number;
  success: number;
  success_rate: number;
  avg_max_return: number;
  avg_final_return: number;
}

interface SixVeinsCombo {
  total: number;
  success: number;
  success_rate: number;
  avg_max_return: number;
  red_count: number;
}

interface SignalSummary {
  scan_date: string;
  total_signals: number;
  success_signals: number;
  overall_success_rate: number;
  holding_days: number;
  success_threshold: number;
  by_signal_type: Record<string, SignalTypeStats>;
  six_veins_combos: Record<string, SixVeinsCombo>;
}

interface IndicatorStats {
  count?: number;
  true_count?: number;
  true_rate?: number;
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  distribution?: Record<string, number>;
}

interface PatternSummary {
  analysis_date: string;
  total_cases: number;
  analyzed_cases: number;
  indicators: Record<string, Record<string, IndicatorStats>>;
  theories: Record<string, Record<string, IndicatorStats>>;
}

interface SignalPatternStats {
  total_cases: number;
  analyzed_cases: number;
  key_patterns: Record<string, IndicatorStats>;
}

// 安全获取数据
const signalSummary = signalSummaryData as SignalSummary;
const patternSummary = patternSummaryData as PatternSummary;
const patternBySignal = patternBySignalData as Record<string, SignalPatternStats>;

export default function AnalysisReport() {
  const [activeTab, setActiveTab] = useState("overview");

  // 按成功率排序的信号类型
  const sortedSignalTypes = Object.entries(signalSummary.by_signal_type || {})
    .sort((a, b) => b[1].success_rate - a[1].success_rate);

  // 按成功率排序的六脉神剑组合（取前15个）
  const sortedCombos = Object.entries(signalSummary.six_veins_combos || {})
    .sort((a, b) => b[1].success_rate - a[1].success_rate)
    .slice(0, 15);

  return (
    <Layout>
      <div className="container py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20">
              <BarChart3 className="w-6 h-6 text-blue-400" />
            </div>
            <h1 className="text-3xl font-bold">分析报告</h1>
          </div>
          <p className="text-muted-foreground">
            信号成功案例分析与模式统计，发现买入信号的共性特征
          </p>
          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
            <span>扫描时间: {signalSummary.scan_date}</span>
            <span>持仓天数: {signalSummary.holding_days}天</span>
            <span>成功阈值: {signalSummary.success_threshold}%</span>
          </div>
        </div>

        {/* 总览卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card className="glass-card bg-gradient-to-br from-blue-500/10 to-blue-600/5">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">总信号数</p>
                  <p className="text-3xl font-bold">{signalSummary.total_signals?.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-full bg-blue-500/20">
                  <Activity className="w-6 h-6 text-blue-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card bg-gradient-to-br from-green-500/10 to-green-600/5">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">成功信号</p>
                  <p className="text-3xl font-bold">{signalSummary.success_signals?.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-full bg-green-500/20">
                  <CheckCircle2 className="w-6 h-6 text-green-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card bg-gradient-to-br from-purple-500/10 to-purple-600/5">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">总体成功率</p>
                  <p className="text-3xl font-bold">{signalSummary.overall_success_rate}%</p>
                </div>
                <div className="p-3 rounded-full bg-purple-500/20">
                  <Target className="w-6 h-6 text-purple-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card bg-gradient-to-br from-orange-500/10 to-orange-600/5">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">分析案例</p>
                  <p className="text-3xl font-bold">{patternSummary.analyzed_cases?.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-full bg-orange-500/20">
                  <PieChart className="w-6 h-6 text-orange-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 标签页 */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid">
            <TabsTrigger value="overview">信号统计</TabsTrigger>
            <TabsTrigger value="sixveins">六脉神剑</TabsTrigger>
            <TabsTrigger value="indicators">指标分析</TabsTrigger>
            <TabsTrigger value="theories">理论分析</TabsTrigger>
          </TabsList>

          {/* 信号统计 */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 按信号类型统计 */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-primary" />
                    按信号类型统计
                  </CardTitle>
                  <CardDescription>各类买入信号的成功率对比</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {sortedSignalTypes.map(([type, stats]) => (
                      <div key={type} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{type}</span>
                          <div className="flex items-center gap-2">
                            <Badge variant={stats.success_rate >= 65 ? "default" : "secondary"}>
                              {stats.success_rate}%
                            </Badge>
                            <span className="text-sm text-muted-foreground">
                              {stats.success}/{stats.total}
                            </span>
                          </div>
                        </div>
                        <Progress value={stats.success_rate} className="h-2" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>平均最大涨幅: {stats.avg_max_return}%</span>
                          <span>平均最终涨幅: {stats.avg_final_return}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* 信号类型详细表格 */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <LineChart className="w-5 h-5 text-primary" />
                    信号详细数据
                  </CardTitle>
                  <CardDescription>各信号类型的详细统计数据</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left py-2 px-2">信号类型</th>
                          <th className="text-right py-2 px-2">总数</th>
                          <th className="text-right py-2 px-2">成功</th>
                          <th className="text-right py-2 px-2">成功率</th>
                          <th className="text-right py-2 px-2">最大涨幅</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedSignalTypes.map(([type, stats]) => (
                          <tr key={type} className="border-b border-white/5 hover:bg-white/5">
                            <td className="py-2 px-2 font-medium">{type}</td>
                            <td className="text-right py-2 px-2">{stats.total.toLocaleString()}</td>
                            <td className="text-right py-2 px-2 text-green-400">{stats.success.toLocaleString()}</td>
                            <td className="text-right py-2 px-2">
                              <span className={stats.success_rate >= 65 ? "text-green-400" : "text-yellow-400"}>
                                {stats.success_rate}%
                              </span>
                            </td>
                            <td className="text-right py-2 px-2">{stats.avg_max_return}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 六脉神剑组合 */}
          <TabsContent value="sixveins" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  六脉神剑组合统计
                </CardTitle>
                <CardDescription>
                  不同指标组合的成功率排名（4红以上）
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="text-left py-3 px-2">排名</th>
                        <th className="text-left py-3 px-2">指标组合</th>
                        <th className="text-center py-3 px-2">红色数</th>
                        <th className="text-right py-3 px-2">总数</th>
                        <th className="text-right py-3 px-2">成功</th>
                        <th className="text-right py-3 px-2">成功率</th>
                        <th className="text-right py-3 px-2">平均涨幅</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedCombos.map(([combo, stats], index) => (
                        <tr key={combo} className="border-b border-white/5 hover:bg-white/5">
                          <td className="py-3 px-2">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                              index < 3 ? "bg-yellow-500/20 text-yellow-400" : "bg-white/10"
                            }`}>
                              {index + 1}
                            </span>
                          </td>
                          <td className="py-3 px-2">
                            <div className="flex flex-wrap gap-1">
                              {combo.split('+').map((indicator) => (
                                <Badge key={indicator} variant="outline" className="text-xs">
                                  {indicator}
                                </Badge>
                              ))}
                            </div>
                          </td>
                          <td className="text-center py-3 px-2">
                            <Badge variant="secondary">{stats.red_count}红</Badge>
                          </td>
                          <td className="text-right py-3 px-2">{stats.total.toLocaleString()}</td>
                          <td className="text-right py-3 px-2 text-green-400">{stats.success.toLocaleString()}</td>
                          <td className="text-right py-3 px-2">
                            <span className={stats.success_rate >= 70 ? "text-green-400 font-semibold" : stats.success_rate >= 60 ? "text-yellow-400" : "text-red-400"}>
                              {stats.success_rate}%
                            </span>
                          </td>
                          <td className="text-right py-3 px-2">{stats.avg_max_return}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 指标分析 */}
          <TabsContent value="indicators" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(patternSummary.indicators || {}).map(([indicator, stats]) => (
                <Card key={indicator} className="glass-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Activity className="w-4 h-4 text-primary" />
                      {indicator}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(stats).map(([field, fieldStats]) => (
                        <div key={field} className="text-sm">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-muted-foreground">{field}</span>
                            {fieldStats.true_rate !== undefined ? (
                              <span className={fieldStats.true_rate >= 60 ? "text-green-400 font-medium" : "text-muted-foreground"}>
                                {fieldStats.true_rate}%
                              </span>
                            ) : fieldStats.mean !== undefined ? (
                              <span className="font-medium">{fieldStats.mean}</span>
                            ) : fieldStats.distribution ? (
                              <span className="text-xs">
                                {Object.entries(fieldStats.distribution).slice(0, 2).map(([k, v]) => `${k}:${v}`).join(', ')}
                              </span>
                            ) : null}
                          </div>
                          {fieldStats.true_rate !== undefined && (
                            <Progress value={fieldStats.true_rate} className="h-1" />
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* 理论分析 */}
          <TabsContent value="theories" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {Object.entries(patternSummary.theories || {}).map(([theory, stats]) => (
                <Card key={theory} className="glass-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-primary" />
                      {theory}
                    </CardTitle>
                    <CardDescription>成功案例的{theory}特征分析</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {Object.entries(stats).map(([field, fieldStats]) => (
                        <div key={field} className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">{field}</span>
                            {fieldStats.true_rate !== undefined ? (
                              <div className="flex items-center gap-1">
                                {fieldStats.true_rate >= 60 ? (
                                  <ArrowUpRight className="w-3 h-3 text-green-400" />
                                ) : (
                                  <ArrowDownRight className="w-3 h-3 text-red-400" />
                                )}
                                <span className={fieldStats.true_rate >= 60 ? "text-green-400 font-medium" : "text-muted-foreground"}>
                                  {fieldStats.true_rate}%
                                </span>
                              </div>
                            ) : fieldStats.mean !== undefined ? (
                              <span className="font-medium">{fieldStats.mean}</span>
                            ) : fieldStats.distribution ? (
                              <div className="text-right">
                                {Object.entries(fieldStats.distribution).slice(0, 3).map(([k, v]) => (
                                  <Badge key={k} variant="outline" className="text-xs ml-1">
                                    {k}: {v}
                                  </Badge>
                                ))}
                              </div>
                            ) : null}
                          </div>
                          {fieldStats.true_rate !== undefined && (
                            <Progress value={fieldStats.true_rate} className="h-1.5" />
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* 按信号类型的模式分析 */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-primary" />
                  按信号类型的关键模式
                </CardTitle>
                <CardDescription>各类信号成功案例的共性特征对比</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="text-left py-3 px-2">信号类型</th>
                        <th className="text-center py-3 px-2">案例数</th>
                        <th className="text-center py-3 px-2">MACD多头</th>
                        <th className="text-center py-3 px-2">KDJ金叉</th>
                        <th className="text-center py-3 px-2">均线多排</th>
                        <th className="text-center py-3 px-2">趋势一致</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(patternBySignal).map(([signalType, data]) => (
                        <tr key={signalType} className="border-b border-white/5 hover:bg-white/5">
                          <td className="py-3 px-2 font-medium">{signalType}</td>
                          <td className="text-center py-3 px-2">{data.analyzed_cases}</td>
                          <td className="text-center py-3 px-2">
                            <span className={(data.key_patterns?.['MACD_DIF>0']?.true_rate || 0) >= 60 ? "text-green-400" : ""}>
                              {data.key_patterns?.['MACD_DIF>0']?.true_rate || '-'}%
                            </span>
                          </td>
                          <td className="text-center py-3 px-2">
                            <span className={(data.key_patterns?.['KDJ_K>D']?.true_rate || 0) >= 60 ? "text-green-400" : ""}>
                              {data.key_patterns?.['KDJ_K>D']?.true_rate || '-'}%
                            </span>
                          </td>
                          <td className="text-center py-3 px-2">
                            <span className={(data.key_patterns?.['均线多头排列']?.true_rate || 0) >= 50 ? "text-green-400" : ""}>
                              {data.key_patterns?.['均线多头排列']?.true_rate || '-'}%
                            </span>
                          </td>
                          <td className="text-center py-3 px-2">
                            <span className={(data.key_patterns?.['道氏趋势一致']?.true_rate || 0) >= 50 ? "text-green-400" : ""}>
                              {data.key_patterns?.['道氏趋势一致']?.true_rate || '-'}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
