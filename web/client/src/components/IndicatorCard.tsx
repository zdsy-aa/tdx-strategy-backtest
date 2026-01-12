import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

interface Signal {
  type: string;
  name: string;
  condition: string;
  action?: string;
  meaning?: string;
  position?: string;
  risk?: string;
  formula?: string;
}

interface SubIndicator {
  name: string;
  formula: string;
  meaning: string;
}

interface IndicatorProps {
  indicator: {
    id: string;
    name: string;
    description: string;
    subIndicators?: SubIndicator[];
    signals: Signal[];
  };
}

export default function IndicatorCard({ indicator }: IndicatorProps) {
  return (
    <Card className="glass-card overflow-hidden transition-all hover:shadow-primary/10 hover:border-primary/30 group">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-bold group-hover:text-primary transition-colors">
            {indicator.name}
          </CardTitle>
          <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
            {indicator.signals.length} 个信号
          </Badge>
        </div>
        <CardDescription className="text-muted-foreground/80">
          {indicator.description}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Sub Indicators Grid */}
        {indicator.subIndicators && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {indicator.subIndicators.map((sub, idx) => (
              <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-sm">{sub.name}</span>
                  <span className="text-xs text-muted-foreground font-mono bg-black/30 px-1.5 py-0.5 rounded">
                    {sub.formula}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">{sub.meaning}</p>
              </div>
            ))}
          </div>
        )}

        {/* Signals List */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">买卖信号</h4>
          {indicator.signals.map((signal, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 rounded-lg bg-black/20 border border-white/5">
              <div className={`mt-0.5 p-1.5 rounded-full ${
                signal.type === 'buy' ? 'bg-green-500/20 text-green-500' :
                signal.type === 'sell' ? 'bg-red-500/20 text-red-500' :
                'bg-yellow-500/20 text-yellow-500'
              }`}>
                {signal.type === 'buy' ? <ArrowUpRight className="size-4" /> :
                 signal.type === 'sell' ? <ArrowDownRight className="size-4" /> :
                 <Minus className="size-4" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className={`font-semibold text-sm ${
                    signal.type === 'buy' ? 'text-green-400' :
                    signal.type === 'sell' ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>
                    {signal.name}
                  </span>
                  {signal.position && (
                    <Badge variant="secondary" className="text-xs h-5">
                      仓位 {signal.position}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mb-1">
                  <span className="text-foreground/80">条件：</span>{signal.condition}
                </p>
                {signal.formula && (
                  <p className="text-xs text-muted-foreground mb-1 font-mono bg-black/30 px-2 py-1 rounded">
                    <span className="text-cyan-400">公式：</span>{signal.formula}
                  </p>
                )}
                {signal.meaning && (
                  <p className="text-xs text-muted-foreground mb-1">
                    <span className="text-yellow-400">含义：</span>{signal.meaning}
                  </p>
                )}
                {signal.risk && (
                  <p className="text-xs text-muted-foreground mb-1">
                    <span className="text-orange-400">风险：</span>{signal.risk}
                  </p>
                )}
                {signal.action && (
                  <p className="text-xs text-muted-foreground">
                    <span className="text-primary/80">操作：</span>{signal.action}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
