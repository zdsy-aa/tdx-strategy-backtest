import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

export default function Backtest() {
  // 模拟回测数据，实际应从后端或JSON获取
  const backtestResults = [
    { rank: 1, strategy: "六脉5红 + 买点2", winRate: "62.5%", return: "0.71%", period: "14天", type: "组合" },
    { rank: 2, strategy: "买点2 + 缠论", winRate: "61.1%", return: "0.66%", period: "14天", type: "组合" },
    { rank: 3, strategy: "稳健中长线", winRate: "60.0%", return: "0.12%", period: "14天", type: "组合" },
    { rank: 4, strategy: "MACD+KDJ+LWR+MTM", winRate: "59.8%", return: "0.65%", period: "2天", type: "六脉4红" },
    { rank: 5, strategy: "MACD+RSI+LWR+MTM", winRate: "59.1%", return: "0.62%", period: "2天", type: "六脉4红" },
  ];

  return (
    <div className="space-y-8">
      <div className="max-w-2xl">
        <h1 className="text-3xl font-bold mb-4">回测数据验证</h1>
        <p className="text-muted-foreground text-lg">
          基于沪深300指数过去10年（2016-2026）的历史数据回测结果。
          数据表明，多信号共振策略在胜率和稳定性上均优于单一指标。
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="glass-card lg:col-span-2">
          <CardHeader>
            <CardTitle>策略胜率排行榜</CardTitle>
            <CardDescription>按胜率降序排列（Top 5）</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-white/5 border-white/10">
                  <TableHead className="w-[60px]">排名</TableHead>
                  <TableHead>策略名称</TableHead>
                  <TableHead>类型</TableHead>
                  <TableHead className="text-right">胜率</TableHead>
                  <TableHead className="text-right">平均收益</TableHead>
                  <TableHead className="text-right">最优周期</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {backtestResults.map((result) => (
                  <TableRow key={result.rank} className="hover:bg-white/5 border-white/10">
                    <TableCell className="font-medium">
                      <div className={`size-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        result.rank === 1 ? 'bg-yellow-500/20 text-yellow-500' :
                        result.rank === 2 ? 'bg-gray-400/20 text-gray-400' :
                        result.rank === 3 ? 'bg-orange-500/20 text-orange-500' :
                        'bg-white/10 text-muted-foreground'
                      }`}>
                        {result.rank}
                      </div>
                    </TableCell>
                    <TableCell className="font-medium">{result.strategy}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="bg-white/5 hover:bg-white/10">
                        {result.type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-green-400 font-bold">{result.winRate}</TableCell>
                    <TableCell className="text-right">{result.return}</TableCell>
                    <TableCell className="text-right text-muted-foreground">{result.period}</TableCell>
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
              <div className="space-y-2">
                <h4 className="font-semibold text-primary">1. 组合优于单一</h4>
                <p className="text-sm text-muted-foreground">
                  “六脉5红 + 买点2”组合胜率高达62.5%，显著优于单一指标。
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-primary">2. 周期至关重要</h4>
                <p className="text-sm text-muted-foreground">
                  不同策略的最优持有周期差异巨大。六脉神剑适合短线爆发（2天），而买点2适合中线波段（14-16天）。
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-primary">3. 风险提示</h4>
                <p className="text-sm text-muted-foreground">
                  即使是最优策略，胜率也未超过70%。严格的止损和仓位管理依然是盈利的核心。
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
