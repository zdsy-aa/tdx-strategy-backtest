import StrategyCard from "@/components/StrategyCard";
import strategiesData from "@/data/strategies.json";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import Layout from "@/components/Layout";

interface SingleIndicatorStrategy {
  id: string;
  name: string;
  indicator: string;
  buyCondition: string;
  suggestedHoldPeriod: string;
  stats: {
    total: {
      winRate: string;
      avgReturn: string;
      optimalPeriod: string;
      trades: number;
    };
    yearly: Record<string, unknown>;
    monthly: Record<string, unknown>;
  };
}

interface SellPointTest {
  indicatorId: string;
  testResults: Array<{
    condition: string;
    winRate: string;
    avgReturn: string;
  }>;
}

export default function Strategies() {
  const allStrategies = strategiesData.strategies;
  const singleIndicatorStrategies = (strategiesData as any).singleIndicatorStrategies as SingleIndicatorStrategy[] || [];
  const sellPointTests = (strategiesData as any).sellPointTests as SellPointTest[] || [];
  const steadyStrategies = allStrategies.filter(s => s.tag === "中长线" || s.tag === "全能");
  const aggressiveStrategies = allStrategies.filter(s => s.tag === "短线" || s.tag === "熊市");

  return (
    <Layout>
      <div className="space-y-8">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-bold mb-4">指标方案</h1>
          <p className="text-muted-foreground text-lg">
            提供单指标买入方案和多指标组合方案。单指标方案适合快速决策，组合方案通过多指标共振提高交易胜率。
            所有方案均提供总数据、年度数据和月度数据的历史胜率统计。
          </p>
        </div>

        <Tabs defaultValue="single" className="w-full">
          <TabsList className="bg-white/5 border border-white/10 p-1 mb-8">
            <TabsTrigger value="single">单指标买入</TabsTrigger>
            <TabsTrigger value="combo">组合方案</TabsTrigger>
            <TabsTrigger value="sellpoint">最佳卖出点</TabsTrigger>
          </TabsList>
          
          {/* 单指标买入方案 */}
          <TabsContent value="single" className="mt-0">
            <div className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>单指标买入方案</CardTitle>
                  <CardDescription>
                    当单一指标触发买入信号时的历史表现统计
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow className="hover:bg-white/5 border-white/10">
                        <TableHead>方案名称</TableHead>
                        <TableHead>所属指标</TableHead>
                        <TableHead>买入条件</TableHead>
                        <TableHead className="text-right">总胜率</TableHead>
                        <TableHead className="text-right">平均收益</TableHead>
                        <TableHead className="text-right">最优周期</TableHead>
                        <TableHead className="text-right">交易次数</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {singleIndicatorStrategies.map((strategy) => (
                        <TableRow key={strategy.id} className="hover:bg-white/5 border-white/10">
                          <TableCell className="font-medium">{strategy.name}</TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="bg-white/5 hover:bg-white/10">
                              {strategy.indicator}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-muted-foreground max-w-[200px] truncate">
                            {strategy.buyCondition}
                          </TableCell>
                          <TableCell className="text-right text-green-400 font-bold">
                            {strategy.stats.total.winRate}
                          </TableCell>
                          <TableCell className="text-right">
                            {strategy.stats.total.avgReturn}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {strategy.stats.total.optimalPeriod}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {strategy.stats.total.trades}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* 单指标详细卡片 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {singleIndicatorStrategies.map((strategy) => (
                  <Card key={strategy.id} className="glass-card hover:border-primary/50 transition-colors">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{strategy.name}</CardTitle>
                        <Badge className="bg-primary/20 text-primary hover:bg-primary/30">
                          {strategy.indicator}
                        </Badge>
                      </div>
                      <CardDescription>{strategy.buyCondition}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <p className="text-xs text-muted-foreground">总胜率</p>
                            <p className="text-2xl font-bold text-green-400">{strategy.stats.total.winRate}</p>
                          </div>
                          <div className="space-y-1">
                            <p className="text-xs text-muted-foreground">平均收益</p>
                            <p className="text-2xl font-bold">{strategy.stats.total.avgReturn}</p>
                          </div>
                        </div>
                        <div className="pt-2 border-t border-white/10">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">最优持有周期</span>
                            <span>{strategy.stats.total.optimalPeriod}</span>
                          </div>
                          <div className="flex justify-between text-sm mt-1">
                            <span className="text-muted-foreground">建议持有</span>
                            <span>{strategy.suggestedHoldPeriod}</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>
          
          {/* 组合方案 */}
          <TabsContent value="combo" className="mt-0">
            <Tabs defaultValue="all" className="w-full">
              <TabsList className="bg-white/5 border border-white/10 p-1 mb-6">
                <TabsTrigger value="all">全部方案</TabsTrigger>
                <TabsTrigger value="steady">稳健/中长线</TabsTrigger>
                <TabsTrigger value="aggressive">激进/短线</TabsTrigger>
              </TabsList>
              
              <TabsContent value="all" className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {allStrategies.map((strategy) => (
                    <StrategyCard key={strategy.id} strategy={strategy} />
                  ))}
                </div>
              </TabsContent>
              
              <TabsContent value="steady" className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {steadyStrategies.map((strategy) => (
                    <StrategyCard key={strategy.id} strategy={strategy} />
                  ))}
                </div>
              </TabsContent>
              
              <TabsContent value="aggressive" className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {aggressiveStrategies.map((strategy) => (
                    <StrategyCard key={strategy.id} strategy={strategy} />
                  ))}
                </div>
              </TabsContent>
            </Tabs>
          </TabsContent>

          {/* 最佳卖出点测试 */}
          <TabsContent value="sellpoint" className="mt-0">
            <div className="space-y-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>单指标最佳卖出点测试</CardTitle>
                  <CardDescription>
                    测试不同卖出条件（固定天数、盈利百分比、指标信号）对收益的影响
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-8">
                  {sellPointTests.map((test) => {
                    const strategy = singleIndicatorStrategies.find(s => s.id === test.indicatorId);
                    return (
                      <div key={test.indicatorId} className="space-y-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                          <Badge className="bg-primary/20 text-primary">
                            {strategy?.name || test.indicatorId}
                          </Badge>
                          卖出点测试结果
                        </h3>
                        <Table>
                          <TableHeader>
                            <TableRow className="hover:bg-white/5 border-white/10">
                              <TableHead>卖出条件</TableHead>
                              <TableHead className="text-right">胜率</TableHead>
                              <TableHead className="text-right">平均收益</TableHead>
                              <TableHead className="text-right">推荐度</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {test.testResults.map((result, idx) => {
                              const winRateNum = parseFloat(result.winRate);
                              const returnNum = parseFloat(result.avgReturn);
                              let recommendation = "一般";
                              let recColor = "text-muted-foreground";
                              if (winRateNum >= 60 && returnNum >= 1) {
                                recommendation = "强烈推荐";
                                recColor = "text-green-400";
                              } else if (winRateNum >= 55 || returnNum >= 2) {
                                recommendation = "推荐";
                                recColor = "text-blue-400";
                              }
                              
                              return (
                                <TableRow key={idx} className="hover:bg-white/5 border-white/10">
                                  <TableCell className="font-medium">{result.condition}</TableCell>
                                  <TableCell className="text-right text-green-400 font-bold">
                                    {result.winRate}
                                  </TableCell>
                                  <TableCell className="text-right">{result.avgReturn}</TableCell>
                                  <TableCell className={`text-right ${recColor}`}>
                                    {recommendation}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    );
                  })}
                </CardContent>
              </Card>

              <Card className="glass-card bg-gradient-to-br from-primary/20 to-purple-600/20 border-primary/20">
                <CardHeader>
                  <CardTitle>卖出策略建议</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold text-primary">1. 固定天数卖出</h4>
                    <p className="text-sm text-muted-foreground">
                      适合纪律性强的投资者。买点2信号的最优持有周期为16天，六脉6红的最优周期为5天左右。
                    </p>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-primary">2. 盈利目标卖出</h4>
                    <p className="text-sm text-muted-foreground">
                      设置3%-5%的盈利目标可以提高胜率，但可能错过更大的涨幅。适合追求稳定收益的投资者。
                    </p>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-primary">3. 信号反转卖出</h4>
                    <p className="text-sm text-muted-foreground">
                      等待卖出信号出现再卖出，可以最大化捕捉趋势。但需要承受更大的回撤风险。
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
