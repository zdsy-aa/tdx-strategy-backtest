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
    const data = backtestData as any;
    
    const total: StrategyTotalStats[] = Object.values(data.strategies).map((s: any) => ({
      name: s.name,
      type: s.type,
      win_rate: s.stats.total.win_rate,
      avg_return: s.stats.total.avg_return,
      best_hold_days: s.stats.total.best_hold_days,
      trades: s.stats.total.trades,
    })).sort((a, b) => b.win_rate - a.win_rate);

    const yearly: StrategyYearlyStats[] = [];
    Object.values(data.strategies).forEach((s: any) => {
        for (const year in s.stats.yearly) {
            const yearData = s.stats.yearly[year];
            yearly.push({
                year: year,
                name: s.name,
                win_rate: yearData.win_rate,
                avg_return: yearData.avg_return,
                trades: yearData.trades
            });
        }
    });
    yearly.sort((a, b) => parseInt(b.year) - parseInt(a.year) || b.win_rate - a.win_rate);

    const monthly: StrategyMonthlyStats[] = Object.values(data.monthly_stats);

    return { totalResults: total, yearlyResults: yearly, monthlyResults: monthly, conclusions: data.conclusions };
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
