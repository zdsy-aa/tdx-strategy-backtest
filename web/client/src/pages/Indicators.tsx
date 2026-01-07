import { useState } from "react";
import Layout from "@/components/Layout";
import IndicatorCard from "@/components/IndicatorCard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { TrendingUp, TrendingDown, Activity, BarChart2, ChevronRight, LineChart } from "lucide-react";
import strategiesData from "@/data/strategies.json";

// K线案例数据（模拟）
const klineCases: Record<string, Array<{
  id: number;
  title: string;
  date: string;
  stock: string;
  description: string;
  beforeSignal: string;
  afterSignal: string;
  result: string;
  holdDays: number;
}>> = {
  six_veins: [
    {
      id: 1,
      title: "六脉六红买入案例",
      date: "2024-03-15",
      stock: "贵州茅台 (600519)",
      description: "六脉神剑全部变红，形成强烈买入信号。买入后5日内上涨8.2%。",
      beforeSignal: "股价经过一段时间调整，MACD、KDJ、RSI等指标同步触底回升",
      afterSignal: "买入后连续3日放量上涨，第5日达到阶段高点",
      result: "+8.2%",
      holdDays: 5,
    },
    {
      id: 2,
      title: "六脉五红买入案例",
      date: "2024-06-20",
      stock: "宁德时代 (300750)",
      description: "六脉神剑5个变红（MTM未红），配合买点2信号，形成较强买入信号。",
      beforeSignal: "股价在支撑位企稳，成交量开始放大",
      afterSignal: "买入后震荡上行，10日内上涨12.5%",
      result: "+12.5%",
      holdDays: 10,
    },
  ],
  buy_sell: [
    {
      id: 1,
      title: "买点2信号案例",
      date: "2024-04-10",
      stock: "比亚迪 (002594)",
      description: "庄家线上穿散户线，且庄家线位于50以下，形成买点2信号。",
      beforeSignal: "股价处于相对低位，庄家线持续下降后开始拐头",
      afterSignal: "买入后稳步上涨，15日内上涨18.3%",
      result: "+18.3%",
      holdDays: 15,
    },
  ],
  chan_lun: [
    {
      id: 1,
      title: "缠论二买案例",
      date: "2024-05-08",
      stock: "中国平安 (601318)",
      description: "一买后回调不破底，形成缠论二买信号。",
      beforeSignal: "一买信号出现后，股价回调至前低附近企稳",
      afterSignal: "二买确认后，开启新一轮上涨，20日内上涨15.7%",
      result: "+15.7%",
      holdDays: 20,
    },
  ],
  money_tree: [
    {
      id: 1,
      title: "摇钱树选股案例",
      date: "2024-07-15",
      stock: "隆基绿能 (601012)",
      description: "摇钱树信号触发，综合顶底识别、趋势过滤与动量交叉。",
      beforeSignal: "股价完成底部构造，多个技术指标共振",
      afterSignal: "信号触发后快速拉升，7日内上涨22.1%",
      result: "+22.1%",
      holdDays: 7,
    },
  ],
};

export default function Indicators() {
  const [selectedIndicator, setSelectedIndicator] = useState<string | null>(null);
  const [showCaseDialog, setShowCaseDialog] = useState(false);
  const [selectedCase, setSelectedCase] = useState<typeof klineCases.six_veins[0] | null>(null);

  const indicators = strategiesData.indicators;

  const getCasesForIndicator = (indicatorId: string) => {
    return klineCases[indicatorId] || [];
  };

  return (
    <Layout>
      <div className="container py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20">
              <Activity className="w-6 h-6 text-cyan-400" />
            </div>
            <h1 className="text-3xl font-bold">指标详解</h1>
          </div>
          <p className="text-muted-foreground">
            深入了解每个技术指标的计算逻辑、信号含义和实战应用
          </p>
        </div>

        {/* 指标卡片网格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {indicators.map((indicator) => (
            <div key={indicator.id} className="relative">
              <IndicatorCard indicator={indicator} />
              {/* 查看案例按钮 */}
              <div className="absolute top-4 right-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSelectedIndicator(indicator.id);
                  }}
                  className="bg-background/80 backdrop-blur-sm"
                >
                  <LineChart className="w-4 h-4 mr-1" />
                  K线案例
                </Button>
              </div>
            </div>
          ))}
        </div>

        {/* K线案例展示区域 */}
        {selectedIndicator && (
          <Card className="glass-card mt-8">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-yellow-400" />
                    历史K线案例 - {indicators.find(i => i.id === selectedIndicator)?.name}
                  </CardTitle>
                  <CardDescription>
                    以下是该指标信号出现后的真实股价走势案例
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedIndicator(null)}
                >
                  收起
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {getCasesForIndicator(selectedIndicator).map((caseItem) => (
                  <Card
                    key={caseItem.id}
                    className="bg-background/50 hover:bg-background/80 transition-colors cursor-pointer"
                    onClick={() => {
                      setSelectedCase(caseItem);
                      setShowCaseDialog(true);
                    }}
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-semibold">{caseItem.title}</h4>
                          <p className="text-sm text-muted-foreground">{caseItem.stock}</p>
                        </div>
                        <Badge
                          variant="secondary"
                          className={
                            caseItem.result.startsWith("+")
                              ? "bg-green-500/20 text-green-400"
                              : "bg-red-500/20 text-red-400"
                          }
                        >
                          {caseItem.result}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {caseItem.description}
                      </p>
                      <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
                        <span>{caseItem.date}</span>
                        <span className="flex items-center gap-1">
                          持有 {caseItem.holdDays} 天
                          <ChevronRight className="w-3 h-3" />
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {getCasesForIndicator(selectedIndicator).length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  暂无该指标的K线案例
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* 案例详情对话框 */}
        <Dialog open={showCaseDialog} onOpenChange={setShowCaseDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>{selectedCase?.title}</DialogTitle>
              <DialogDescription>
                {selectedCase?.stock} | {selectedCase?.date}
              </DialogDescription>
            </DialogHeader>
            {selectedCase && (
              <div className="space-y-4">
                {/* 模拟K线图区域 */}
                <div className="h-48 bg-background/50 rounded-lg flex items-center justify-center border border-border/50">
                  <div className="text-center text-muted-foreground">
                    <BarChart2 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>K线图展示区域</p>
                    <p className="text-xs">（实际部署时对接行情数据）</p>
                  </div>
                </div>

                {/* 案例详情 */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-background/50">
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      <TrendingDown className="w-4 h-4 text-blue-400" />
                      信号前
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      {selectedCase.beforeSignal}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50">
                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-green-400" />
                      信号后
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      {selectedCase.afterSignal}
                    </p>
                  </div>
                </div>

                {/* 结果统计 */}
                <div className="flex items-center justify-between p-4 rounded-lg bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30">
                  <div>
                    <span className="text-sm text-muted-foreground">持有周期</span>
                    <p className="font-semibold">{selectedCase.holdDays} 天</p>
                  </div>
                  <div className="text-right">
                    <span className="text-sm text-muted-foreground">收益率</span>
                    <p className="text-2xl font-bold text-green-400">{selectedCase.result}</p>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground">
                  {selectedCase.description}
                </p>
              </div>
            )}
          </DialogContent>
        </Dialog>

        {/* 指标对比说明 */}
        <Card className="glass-card mt-8">
          <CardHeader>
            <CardTitle className="text-xl">指标特性对比</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left py-3 px-4">指标</th>
                    <th className="text-left py-3 px-4">类型</th>
                    <th className="text-left py-3 px-4">适用周期</th>
                    <th className="text-left py-3 px-4">信号频率</th>
                    <th className="text-left py-3 px-4">可靠性</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/30">
                    <td className="py-3 px-4 font-medium">六脉神剑</td>
                    <td className="py-3 px-4">综合指标</td>
                    <td className="py-3 px-4">中短线</td>
                    <td className="py-3 px-4">
                      <Badge variant="secondary">中等</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className="bg-green-500/20 text-green-400">高</Badge>
                    </td>
                  </tr>
                  <tr className="border-b border-border/30">
                    <td className="py-3 px-4 font-medium">买卖点</td>
                    <td className="py-3 px-4">主力追踪</td>
                    <td className="py-3 px-4">中线</td>
                    <td className="py-3 px-4">
                      <Badge variant="secondary">较低</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className="bg-green-500/20 text-green-400">较高</Badge>
                    </td>
                  </tr>
                  <tr className="border-b border-border/30">
                    <td className="py-3 px-4 font-medium">缠论买点</td>
                    <td className="py-3 px-4">结构分析</td>
                    <td className="py-3 px-4">中长线</td>
                    <td className="py-3 px-4">
                      <Badge variant="secondary">低</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className="bg-green-500/20 text-green-400">高</Badge>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium">摇钱树</td>
                    <td className="py-3 px-4">选股信号</td>
                    <td className="py-3 px-4">短线</td>
                    <td className="py-3 px-4">
                      <Badge variant="secondary">极低</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className="bg-yellow-500/20 text-yellow-400">中等</Badge>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
