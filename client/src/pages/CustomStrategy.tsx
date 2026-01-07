import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Loader2, Plus, Trash2, Play, Settings2, BarChart3 } from "lucide-react";
import { trpc } from "@/lib/trpc";

interface Condition {
  id: string;
  indicator: string;
  operator: "eq" | "gt" | "lt" | "gte" | "lte";
  value: number | boolean;
}

export default function CustomStrategy() {
  const [strategyName, setStrategyName] = useState("");
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [holdPeriod, setHoldPeriod] = useState(5);
  const [offsetDays, setOffsetDays] = useState(5);
  const [result, setResult] = useState<any>(null);

  const { data: indicators } = trpc.backtest.indicators.useQuery();

  const backtestMutation = trpc.backtest.run.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const addCondition = () => {
    setConditions([
      ...conditions,
      {
        id: Date.now().toString(),
        indicator: "",
        operator: "eq",
        value: true,
      },
    ]);
  };

  const removeCondition = (id: string) => {
    setConditions(conditions.filter((c) => c.id !== id));
  };

  const updateCondition = (id: string, field: keyof Condition, value: any) => {
    setConditions(
      conditions.map((c) => (c.id === id ? { ...c, [field]: value } : c))
    );
  };

  const handleBacktest = () => {
    if (!strategyName.trim() || conditions.length === 0) return;

    const validConditions = conditions
      .filter((c) => c.indicator)
      .map((c) => ({
        indicator: c.indicator,
        operator: c.operator,
        value: c.value,
      }));

    if (validConditions.length === 0) return;

    backtestMutation.mutate({
      name: strategyName,
      conditions: validConditions,
      holdPeriod,
      offsetDays,
    });
  };

  const operatorLabels: Record<string, string> = {
    eq: "等于",
    gt: "大于",
    lt: "小于",
    gte: "大于等于",
    lte: "小于等于",
  };

  return (
    <Layout>
      <div className="container py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20">
              <Settings2 className="w-6 h-6 text-blue-400" />
            </div>
            <h1 className="text-3xl font-bold">自定义策略</h1>
          </div>
          <p className="text-muted-foreground">
            自由组合不同的指标信号，构建并回测您的专属交易策略
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 左侧：策略构建器 */}
          <div className="space-y-6">
            {/* 基本信息 */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">策略基本信息</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">策略名称</label>
                  <Input
                    placeholder="例如：六脉五红+买点2组合"
                    value={strategyName}
                    onChange={(e) => setStrategyName(e.target.value)}
                    className="bg-background/50"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">持有周期 ({holdPeriod}天)</label>
                    <Slider
                      value={[holdPeriod]}
                      onValueChange={([v]) => setHoldPeriod(v)}
                      min={1}
                      max={30}
                      step={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">信号偏移 ({offsetDays}天)</label>
                    <Slider
                      value={[offsetDays]}
                      onValueChange={([v]) => setOffsetDays(v)}
                      min={0}
                      max={10}
                      step={1}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 条件构建器 */}
            <Card className="glass-card">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">买入条件</CardTitle>
                    <CardDescription>添加指标条件，满足所有条件时触发买入信号</CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={addCondition}>
                    <Plus className="w-4 h-4 mr-1" />
                    添加条件
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {conditions.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    点击"添加条件"开始构建策略
                  </div>
                ) : (
                  conditions.map((condition, index) => (
                    <div
                      key={condition.id}
                      className="flex items-center gap-2 p-3 rounded-lg bg-background/50"
                    >
                      <span className="text-sm text-muted-foreground w-6">
                        {index + 1}.
                      </span>
                      <Select
                        value={condition.indicator}
                        onValueChange={(v) => updateCondition(condition.id, "indicator", v)}
                      >
                        <SelectTrigger className="w-[180px]">
                          <SelectValue placeholder="选择指标" />
                        </SelectTrigger>
                        <SelectContent>
                          {indicators?.map((ind) => (
                            <SelectItem key={ind.id} value={ind.id}>
                              {ind.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select
                        value={condition.operator}
                        onValueChange={(v) => updateCondition(condition.id, "operator", v)}
                      >
                        <SelectTrigger className="w-[120px]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(operatorLabels).map(([key, label]) => (
                            <SelectItem key={key} value={key}>
                              {label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Input
                        type="number"
                        value={typeof condition.value === "boolean" ? (condition.value ? 1 : 0) : condition.value}
                        onChange={(e) => updateCondition(condition.id, "value", Number(e.target.value))}
                        className="w-[80px] bg-background"
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removeCondition(condition.id)}
                        className="text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            {/* 执行按钮 */}
            <Button
              onClick={handleBacktest}
              disabled={!strategyName.trim() || conditions.length === 0 || backtestMutation.isPending}
              className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
              size="lg"
            >
              {backtestMutation.isPending ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  回测中...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  执行回测
                </>
              )}
            </Button>
          </div>

          {/* 右侧：回测结果 */}
          <div className="space-y-6">
            {result ? (
              <>
                <Card className="glass-card border-green-500/30">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-green-400" />
                      回测结果
                    </CardTitle>
                    <CardDescription>
                      策略: {result.strategyName} | 数据范围: {result.dataRange.start} 至 {result.dataRange.end}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">信号次数</div>
                        <div className="text-2xl font-bold">{result.results.signalCount}</div>
                      </div>
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">交易次数</div>
                        <div className="text-2xl font-bold">{result.results.tradeCount}</div>
                      </div>
                      <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
                        <div className="text-sm text-muted-foreground">胜率</div>
                        <div className="text-2xl font-bold text-green-400">{result.results.winRate}%</div>
                      </div>
                      <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
                        <div className="text-sm text-muted-foreground">平均收益</div>
                        <div className="text-2xl font-bold text-blue-400">{result.results.avgReturn}%</div>
                      </div>
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">最大收益</div>
                        <div className="text-xl font-bold text-green-400">+{result.results.maxReturn}%</div>
                      </div>
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">最大亏损</div>
                        <div className="text-xl font-bold text-red-400">{result.results.minReturn}%</div>
                      </div>
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">累计收益</div>
                        <div className="text-xl font-bold">{result.results.totalReturn}%</div>
                      </div>
                      <div className="p-4 rounded-lg bg-background/50">
                        <div className="text-sm text-muted-foreground">夏普比率</div>
                        <div className="text-xl font-bold">{result.results.sharpeRatio}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="text-lg">策略条件</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {result.conditions.map((c: any, i: number) => (
                        <Badge key={i} variant="secondary">
                          {indicators?.find((ind) => ind.id === c.indicator)?.name || c.indicator}{" "}
                          {operatorLabels[c.operator]} {c.value}
                        </Badge>
                      ))}
                    </div>
                    <div className="mt-4 text-sm text-muted-foreground">
                      持有周期: {result.holdPeriod}天 | 信号偏移: {result.offsetDays}天
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="glass-card">
                <CardContent className="py-16 text-center text-muted-foreground">
                  <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>构建策略并执行回测后，结果将显示在这里</p>
                </CardContent>
              </Card>
            )}

            {/* 预设策略 */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">快速预设</CardTitle>
                <CardDescription>点击加载预设策略条件</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => {
                    setStrategyName("六脉五红+买点2");
                    setConditions([
                      { id: "1", indicator: "six_veins_count", operator: "gte", value: 5 },
                      { id: "2", indicator: "buy2", operator: "eq", value: 1 },
                    ]);
                    setHoldPeriod(5);
                  }}
                >
                  六脉五红 + 买点2
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => {
                    setStrategyName("六脉六红");
                    setConditions([
                      { id: "1", indicator: "six_veins_count", operator: "eq", value: 6 },
                    ]);
                    setHoldPeriod(3);
                  }}
                >
                  六脉六红 (激进)
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => {
                    setStrategyName("缠论二买+六脉四红");
                    setConditions([
                      { id: "1", indicator: "chan_buy2", operator: "eq", value: 1 },
                      { id: "2", indicator: "six_veins_count", operator: "gte", value: 4 },
                    ]);
                    setHoldPeriod(10);
                  }}
                >
                  缠论二买 + 六脉四红 (稳健)
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </Layout>
  );
}
