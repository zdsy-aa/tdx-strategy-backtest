import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CheckCircle2, XCircle, Clock, TrendingUp, AlertTriangle } from "lucide-react";

interface StrategyStats {
  total: {
    winRate: string;
    avgReturn: string;
    optimalPeriod: string;
    trades: number;
  };
  yearly: Record<string, unknown>;
  monthly: Record<string, unknown>;
}

interface StrategyProps {
  strategy: {
    id: string;
    name: string;
    tag: string;
    period: string;
    buyConditions: string[];
    sellConditions: string[];
    winRate?: string;
    avgReturn?: string;
    stats?: StrategyStats;
  };
}

export default function StrategyCard({ strategy }: StrategyProps) {
  // 支持旧格式和新格式
  const winRate = strategy.stats?.total?.winRate || strategy.winRate || "待测";
  const avgReturn = strategy.stats?.total?.avgReturn || strategy.avgReturn || "待测";
  const trades = strategy.stats?.total?.trades || 0;
  
  const winRateNum = parseFloat(winRate);
  const isHighWinRate = !isNaN(winRateNum) && winRateNum > 60;

  return (
    <Card className="glass-card flex flex-col h-full hover:scale-[1.02] transition-transform duration-300 relative overflow-hidden">
      {/* Background Gradient Blob */}
      <div className={`absolute -top-20 -right-20 w-40 h-40 rounded-full blur-3xl opacity-20 ${
        strategy.id === 'aggressive' ? 'bg-red-500' :
        strategy.id === 'steady' ? 'bg-blue-500' :
        strategy.id === 'resonance' ? 'bg-purple-500' :
        'bg-green-500'
      }`} />

      <CardHeader>
        <div className="flex items-center justify-between mb-2">
          <Badge variant="outline" className={`
            ${strategy.id === 'aggressive' ? 'border-red-500/50 text-red-400 bg-red-500/10' :
              strategy.id === 'steady' ? 'border-blue-500/50 text-blue-400 bg-blue-500/10' :
              strategy.id === 'resonance' ? 'border-purple-500/50 text-purple-400 bg-purple-500/10' :
              'border-green-500/50 text-green-400 bg-green-500/10'}
          `}>
            {strategy.tag}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="size-3" />
            {strategy.period}
          </div>
        </div>
        <CardTitle className="text-2xl">{strategy.name}</CardTitle>
      </CardHeader>

      <CardContent className="flex-1 space-y-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-black/30 border border-white/5 text-center">
            <div className="text-xs text-muted-foreground mb-1">总胜率</div>
            <div className={`text-xl font-bold ${isHighWinRate ? 'text-green-400' : 'text-foreground'}`}>
              {winRate}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-black/30 border border-white/5 text-center">
            <div className="text-xs text-muted-foreground mb-1">平均收益</div>
            <div className="text-xl font-bold text-primary">
              {avgReturn}
            </div>
          </div>
        </div>

        {/* Trade Count */}
        {trades > 0 && (
          <div className="text-center text-sm text-muted-foreground">
            历史交易次数: <span className="text-foreground font-medium">{trades}</span>
          </div>
        )}

        {/* Conditions */}
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-2">
              <CheckCircle2 className="size-4" /> 买入条件
            </h4>
            <ul className="space-y-1.5">
              {strategy.buyConditions.map((cond, idx) => (
                <li key={idx} className="text-sm text-muted-foreground pl-6 relative before:absolute before:left-2 before:top-2 before:size-1.5 before:rounded-full before:bg-green-500/50">
                  {cond}
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
              <XCircle className="size-4" /> 卖出条件
            </h4>
            <ul className="space-y-1.5">
              {strategy.sellConditions.map((cond, idx) => (
                <li key={idx} className="text-sm text-muted-foreground pl-6 relative before:absolute before:left-2 before:top-2 before:size-1.5 before:rounded-full before:bg-red-500/50">
                  {cond}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>

      <CardFooter className="pt-4 border-t border-white/5">
        <Button className="w-full bg-white/5 hover:bg-white/10 text-foreground border border-white/10">
          查看详细回测报告
        </Button>
      </CardFooter>
    </Card>
  );
}
