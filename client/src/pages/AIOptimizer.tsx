import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2, Sparkles, Brain, TrendingUp, Shield, Zap } from "lucide-react";
import { Streamdown } from "streamdown";
import { trpc } from "@/lib/trpc";

export default function AIOptimizer() {
  const [query, setQuery] = useState("");
  const [riskLevel, setRiskLevel] = useState<"conservative" | "moderate" | "aggressive" | undefined>();
  const [holdPeriod, setHoldPeriod] = useState<"short" | "medium" | "long" | undefined>();
  const [marketCondition, setMarketCondition] = useState<"bull" | "bear" | "range" | undefined>();
  const [result, setResult] = useState<string | null>(null);

  const optimizeMutation = trpc.ai.optimize.useMutation({
    onSuccess: (data) => {
      setResult(data.suggestion);
    },
  });

  const handleOptimize = () => {
    if (!query.trim()) return;
    
    optimizeMutation.mutate({
      query,
      context: {
        riskLevel,
        holdPeriod,
        marketCondition,
      },
    });
  };

  const exampleQueries = [
    "我想在牛市中找到短线机会，风险承受能力较高",
    "帮我设计一个适合震荡市的稳健策略",
    "如何利用六脉神剑和买点2进行组合操作？",
    "熊市中如何识别抄底机会？",
  ];

  return (
    <Layout>
      <div className="container py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20">
              <Brain className="w-6 h-6 text-purple-400" />
            </div>
            <h1 className="text-3xl font-bold">AI 策略优化</h1>
          </div>
          <p className="text-muted-foreground">
            通过自然语言描述您的交易需求，AI 将为您推荐最适合的策略组合
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：输入区域 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 上下文设置 */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Shield className="w-5 h-5 text-blue-400" />
                  交易偏好设置
                </CardTitle>
                <CardDescription>
                  设置您的风险偏好和市场判断，帮助 AI 给出更精准的建议
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">风险偏好</label>
                    <Select value={riskLevel} onValueChange={(v) => setRiskLevel(v as any)}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择风险等级" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="conservative">稳健型</SelectItem>
                        <SelectItem value="moderate">平衡型</SelectItem>
                        <SelectItem value="aggressive">激进型</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">持有周期</label>
                    <Select value={holdPeriod} onValueChange={(v) => setHoldPeriod(v as any)}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择持有周期" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="short">短线 (1-5天)</SelectItem>
                        <SelectItem value="medium">中线 (5-15天)</SelectItem>
                        <SelectItem value="long">长线 (15天以上)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">市场环境</label>
                    <Select value={marketCondition} onValueChange={(v) => setMarketCondition(v as any)}>
                      <SelectTrigger>
                        <SelectValue placeholder="选择市场环境" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="bull">牛市</SelectItem>
                        <SelectItem value="bear">熊市</SelectItem>
                        <SelectItem value="range">震荡市</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 输入框 */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-400" />
                  描述您的需求
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="例如：我想在牛市中找到短线机会，风险承受能力较高，请推荐适合的指标组合..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="min-h-[120px] bg-background/50"
                />
                <div className="flex justify-between items-center">
                  <div className="text-sm text-muted-foreground">
                    {query.length}/2000 字符
                  </div>
                  <Button 
                    onClick={handleOptimize}
                    disabled={!query.trim() || optimizeMutation.isPending}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                  >
                    {optimizeMutation.isPending ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        分析中...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 mr-2" />
                        获取建议
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* AI 回复 */}
            {result && (
              <Card className="glass-card border-purple-500/30">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-400" />
                    AI 策略建议
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-invert max-w-none">
                    <Streamdown>{result}</Streamdown>
                  </div>
                </CardContent>
              </Card>
            )}

            {optimizeMutation.isError && (
              <Card className="glass-card border-red-500/30">
                <CardContent className="pt-6">
                  <p className="text-red-400">
                    获取建议失败：{optimizeMutation.error.message}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* 右侧：示例和提示 */}
          <div className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">示例问题</CardTitle>
                <CardDescription>点击下方示例快速开始</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {exampleQueries.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setQuery(example)}
                    className="w-full text-left p-3 rounded-lg bg-background/50 hover:bg-background/80 transition-colors text-sm"
                  >
                    {example}
                  </button>
                ))}
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">可用指标</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">六脉神剑</Badge>
                  <Badge variant="secondary">买点1/2</Badge>
                  <Badge variant="secondary">卖点1/2</Badge>
                  <Badge variant="secondary">缠论买点</Badge>
                  <Badge variant="secondary">摇钱树</Badge>
                  <Badge variant="secondary">MACD</Badge>
                  <Badge variant="secondary">KDJ</Badge>
                  <Badge variant="secondary">RSI</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card bg-gradient-to-br from-blue-500/10 to-purple-500/10">
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-2">💡 使用提示</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• 描述越详细，建议越精准</li>
                  <li>• 可以询问特定指标的用法</li>
                  <li>• 可以请求组合策略的优化</li>
                  <li>• 支持中文自然语言交互</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </Layout>
  );
}
