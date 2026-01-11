import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import backtestData from "@/data/backtest_results.json";
import { useMemo } from "react";

interface StrategyTotalStats {
  name: string;
  type: string;
  win_rate: number;
  avg_return: number;
  best_hold_days: number;
  trades: number;
}

interface StrategyYearlyStats {
  year: string;
  name: string;
  win_rate: number;
  avg_return: number;
  trades: number;
}

interface StrategyMonthlyStats {
  month: string;
  avg_win_rate: number;
  avg_return: number;
  best_strategy: string;
}

export default function Backtest() {
  const { totalResults, yearlyResults, monthlyResults, conclusions } = useMemo(() => {
    const data = backtestData as any[];
    
    if (!Array.isArray(data)) {
      return { totalResults: [], yearlyResults: [], monthlyResults: [], conclusions: [] };
    }

    const parsePct = (val: string) => parseFloat(val.replace('%', '')) || 0;

    const total: StrategyTotalStats[] = data.map((s: any) => ({
      name: s.name,
      type: s.id.startsWith('combo_') ? "组合" : "单指标",
      win_rate: parsePct(s.total.win_rate),
      avg_return: parsePct(s.total.avg_return),
      best_hold_days: parseInt(s.optimal_period_win) || 5,
      trades: s.total.trades,
    })).sort((a, b) => b.win_rate - a.win_rate);

    const yearly: StrategyYearlyStats[] = [];
    data.forEach((s: any) => {
        for (const year in s.yearly) {
            const yearData = s.yearly[year];
            yearly.push({
                year: year,
                name: s.name,
                win_rate: parsePct(yearData.win_rate),
                avg_return: parsePct(yearData.avg_return),
                trades: yearData.trades
            });
        }
    });
    yearly.sort((a, b) => parseInt(b.year) - parseInt(a.year) || b.win_rate - a.win_rate);

    // 计算月度统计
    const monthlyMap = new Map<string, { win_rates: number[], returns: number[], strategies: {name: string, win_rate: number}[] }>();
    data.forEach((s: any) => {
      for (const month in s.monthly) {
        const mData = s.monthly[month];
        if (!monthlyMap.has(month)) {
          monthlyMap.set(month, { win_rates: [], returns: [], strategies: [] });
        }
        const m = monthlyMap.get(month)!;
        const wr = parsePct(mData.win_rate);
        m.win_rates.push(wr);
        m.returns.push(parsePct(mData.avg_return));
        m.strategies.push({ name: s.name, win_rate: wr });
      }
    });

    const monthly: StrategyMonthlyStats[] = Array.from(monthlyMap.entries()).map(([month, m]) => ({
      month,
      avg_win_rate: m.win_rates.reduce((a, b) => a + b, 0) / m.win_rates.length,
      avg_return: m.returns.reduce((a, b) => a + b, 0) / m.returns.length,
      best_strategy: m.strategies.sort((a, b) => b.win_rate - a.win_rate)[0]?.name || "-"
    })).sort((a, b) => {
      const ma = parseInt(a.month.replace('月', ''));
      const mb = parseInt(b.month.replace('月', ''));
      return ma - mb;
    });

    const defaultConclusions = [
      "高胜率策略：六脉神剑系列在短线交易中表现出极高的胜率稳定性。",
      "最优持有期：大多数策略在 5-15 个交易日内能达到收益最大化。",
      "风险提示：回测数据基于历史表现，不代表未来收益，请谨慎参考。"
    ];

    return { totalResults: total, yearlyResults: yearly, monthlyResults: monthly, conclusions: defaultConclusions };
  }, []);

  return (
    <Layout>
      <div className="container py-8 space-y-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">回测数据验证</h1>
          <p className="text-muted-foreground">
            基于历史数据的回测结果，分为总数据、年度数据和月度数据三个维度。
          </p>
        </div>

        <Tabs defaultValue="total" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="total">总数据</TabsTrigger>
            <TabsTrigger value="yearly">年度数据</TabsTrigger>
            <TabsTrigger value="monthly">月度数据</TabsTrigger>
          </TabsList>

          {/* 总数据 */}
          <TabsContent value="total">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
              <Card className="glass-card lg:col-span-2">
                <CardHeader>
                  <CardTitle>策略胜率排行榜（总数据）</CardTitle>
                  <CardDescription>基于全部历史数据，按胜率降序排列</CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow className="hover:bg-white/5 border-white/10">
                        <TableHead className="w-[50px]">排名</TableHead>
                        <TableHead>策略名称</TableHead>
                        <TableHead>类型</TableHead>
                        <TableHead className="text-right">胜率</TableHead>
                        <TableHead className="text-right">平均收益</TableHead>
                        <TableHead>最优周期</TableHead>
                        <TableHead className="text-right">交易次数</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {totalResults.map((result, index) => (
                        <TableRow key={result.name} className="hover:bg-white/5 border-white/10">
                           <TableCell>
                             <div className={`size-8 rounded-full flex items-center justify-center text-xs font-bold ${
                               index === 0 ? 'bg-yellow-500/20 text-yellow-500' :
                               index === 1 ? 'bg-gray-400/20 text-gray-400' :
                               index === 2 ? 'bg-orange-500/20 text-orange-500' :
                               'bg-white/10 text-muted-foreground'
                             }`}>
                               {index + 1}
                             </div>
                           </TableCell>
                          <TableCell className="font-medium">{result.name}</TableCell>
                          <TableCell>
                            <Badge variant={result.type === "组合" ? "default" : "secondary"}>{result.type}</Badge>
                          </TableCell>
                          <TableCell className="text-right text-green-400 font-bold">{result.win_rate.toFixed(1)}%</TableCell>
                          <TableCell className="text-right">{result.avg_return.toFixed(2)}%</TableCell>
                          <TableCell>{result.best_hold_days}天</TableCell>
                          <TableCell className="text-right">{result.trades}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <div className="space-y-6">
                <Card className="glass-card bg-gradient-to-br from-primary/20 to-purple-600/20 border-primary/20">
                  <CardHeader>
                    <CardTitle>回测结论</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {conclusions.map((conclusion: string, index: number) => (
                       <div className="space-y-2" key={index}>
                         <h4 className="font-semibold text-primary">{index + 1}. {conclusion.split("：")[0]}</h4>
                         <p className="text-sm text-muted-foreground">
                           {conclusion.split("：")[1]}
                         </p>
                       </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* 年度数据 */}
          <TabsContent value="yearly">
            <Card className="glass-card mt-6">
              <CardHeader>
                <CardTitle>年度回测数据</CardTitle>
                <CardDescription>按年份分组的策略表现统计</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-white/5 border-white/10">
                      <TableHead>年份</TableHead>
                      <TableHead>策略名称</TableHead>
                      <TableHead className="text-right">胜率</TableHead>
                      <TableHead className="text-right">平均收益</TableHead>
                      <TableHead className="text-right">交易次数</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {yearlyResults.map((result, idx) => (
                      <TableRow key={idx} className="hover:bg-white/5 border-white/10">
                        <TableCell className="font-medium">
                          <Badge variant="outline" className="bg-white/5">
                            {result.year}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-medium">{result.name}</TableCell>
                        <TableCell className="text-right text-green-400 font-bold">{result.win_rate.toFixed(1)}%</TableCell>
                        <TableCell className="text-right">{result.avg_return.toFixed(2)}%</TableCell>
                        <TableCell className="text-right text-muted-foreground">{result.trades}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* 月度数据 */}
          <TabsContent value="monthly">
            <Card className="glass-card mt-6">
              <CardHeader>
                <CardTitle>月度回测数据</CardTitle>
                <CardDescription>按月份分组的策略表现统计，帮助识别季节性规律</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-white/5 border-white/10">
                      <TableHead>月份</TableHead>
                      <TableHead className="text-right">平均胜率</TableHead>
                      <TableHead className="text-right">平均收益</TableHead>
                      <TableHead>最佳策略</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {monthlyResults.map((result, idx) => (
                      <TableRow key={idx} className="hover:bg-white/5 border-white/10">
                        <TableCell className="font-medium">{result.month}</TableCell>
                        <TableCell className="text-right font-bold text-green-400">{result.avg_win_rate.toFixed(1)}%</TableCell>
                        <TableCell className="text-right">{result.avg_return.toFixed(2)}%</TableCell>
                        <TableCell>
                          <Badge variant="secondary">{result.best_strategy}</Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
