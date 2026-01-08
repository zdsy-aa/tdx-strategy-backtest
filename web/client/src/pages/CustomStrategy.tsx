import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Plus, Trash2, Play, Settings2, AlertCircle } from "lucide-react";

interface Condition {
  id: string;
  indicator: string;
  operator: "eq" | "gt" | "lt" | "gte" | "lte";
  value: number | boolean;
}

// 静态指标列表
const staticIndicators = [
  { id: "six_veins_6red", name: "六脉6红" },
  { id: "six_veins_5red", name: "六脉5红" },
  { id: "buy_point_1", name: "买点1" },
  { id: "buy_point_2", name: "买点2" },
  { id: "sell_point_1", name: "卖点1" },
  { id: "sell_point_2", name: "卖点2" },
  { id: "chan_buy_1", name: "缠论一买" },
  { id: "chan_buy_2", name: "缠论二买" },
  { id: "money_tree", name: "摇钱树" },
  { id: "macd_golden", name: "MACD金叉" },
  { id: "kdj_golden", name: "KDJ金叉" },
];

export default function CustomStrategy() {
  const [strategyName, setStrategyName] = useState("");
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [holdPeriod, setHoldPeriod] = useState(5);
  const [offsetDays, setOffsetDays] = useState(5);

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
            {/* 功能提示 */}
            <Card className="glass-card border-yellow-500/30">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-yellow-400 mb-1">静态部署模式</h3>
                    <p className="text-sm text-muted-foreground">
                      当前网站为静态部署版本，自定义策略回测功能需要后端服务支持。
                      如需使用此功能，请在本地运行完整版本（pnpm dev）。
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

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
                          {staticIndicators.map((ind) => (
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
              disabled={true}
              className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 opacity-50"
              size="lg"
            >
              <Play className="w-5 h-5 mr-2" />
              执行回测（需后端支持）
            </Button>
          </div>

          {/* 右侧：说明 */}
          <div className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">功能说明</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">
                  自定义策略功能允许您自由组合多个技术指标，构建个性化的交易策略并进行历史回测验证。
                </p>
                <div className="space-y-2">
                  <h4 className="font-semibold">使用步骤：</h4>
                  <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-1">
                    <li>输入策略名称</li>
                    <li>设置持有周期和信号偏移</li>
                    <li>添加买入条件（可多个）</li>
                    <li>点击执行回测查看结果</li>
                  </ol>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">可用指标</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {staticIndicators.map((ind) => (
                    <Badge key={ind.id} variant="secondary">{ind.name}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card bg-gradient-to-br from-blue-500/10 to-cyan-500/10">
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">💡 策略构建技巧</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• 多指标共振可提高信号可靠性</li>
                  <li>• 短周期适合短线，长周期适合波段</li>
                  <li>• 信号偏移可以避免追高买入</li>
                  <li>• 建议先用少量条件测试</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </Layout>
  );
}
