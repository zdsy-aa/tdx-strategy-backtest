import { useState } from "react";
import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { FileText, Calendar, TrendingUp, ChevronRight } from "lucide-react";
import backtestData from "@/data/backtest_results.json";

// 静态报告数据
const staticReports = [
  {
    id: "total/overall_summary",
    name: "策略总览报告",
    type: "total" as const,
    path: "total/overall_summary",
    updatedAt: new Date().toISOString(),
    content: generateOverallReport(),
  },
  {
    id: "yearly/2025",
    name: "2025年度报告",
    type: "yearly" as const,
    path: "yearly/2025",
    updatedAt: new Date().toISOString(),
    content: generateYearlyReport("2025"),
  },
  {
    id: "yearly/2024",
    name: "2024年度报告",
    type: "yearly" as const,
    path: "yearly/2024",
    updatedAt: new Date().toISOString(),
    content: generateYearlyReport("2024"),
  },
  {
    id: "yearly/2023",
    name: "2023年度报告",
    type: "yearly" as const,
    path: "yearly/2023",
    updatedAt: new Date().toISOString(),
    content: generateYearlyReport("2023"),
  },
];

function generateOverallReport() {
  const strategies = (backtestData as any).strategies || {};
  let report = "# 策略总览报告\n\n";
  report += "## 各策略总体表现\n\n";
  report += "| 策略名称 | 交易次数 | 胜率 | 平均收益 | 总收益 |\n";
  report += "|---------|---------|------|---------|-------|\n";
  
  for (const [key, value] of Object.entries(strategies)) {
    const s = value as any;
    const total = s.stats?.total || {};
    report += `| ${s.name} | ${total.trades || 0} | ${total.win_rate || 0}% | ${total.avg_return || 0}% | ${total.total_return || 0}% |\n`;
  }
  
  return report;
}

function generateYearlyReport(year: string) {
  const strategies = (backtestData as any).strategies || {};
  let report = `# ${year}年度回测报告\n\n`;
  report += "## 各策略年度表现\n\n";
  report += "| 策略名称 | 交易次数 | 胜率 | 平均收益 |\n";
  report += "|---------|---------|------|--------|\n";
  
  for (const [key, value] of Object.entries(strategies)) {
    const s = value as any;
    const yearly = s.stats?.yearly?.[year] || {};
    if (yearly.trades) {
      report += `| ${s.name} | ${yearly.trades} | ${yearly.win_rate}% | ${yearly.avg_return}% |\n`;
    }
  }
  
  return report;
}

export default function Reports() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("total");

  const filteredReports = staticReports.filter((r) => r.type === activeTab);
  const selectedReportData = staticReports.find((r) => r.path === selectedReport);

  const typeLabels: Record<string, string> = {
    total: "总报告",
    yearly: "年度报告",
    monthly: "月度报告",
  };

  const typeIcons: Record<string, React.ReactNode> = {
    total: <TrendingUp className="w-4 h-4" />,
    yearly: <Calendar className="w-4 h-4" />,
    monthly: <FileText className="w-4 h-4" />,
  };

  return (
    <Layout>
      <div className="container py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-orange-500/20 to-red-500/20">
              <FileText className="w-6 h-6 text-orange-400" />
            </div>
            <h1 className="text-3xl font-bold">回测报告</h1>
          </div>
          <p className="text-muted-foreground">
            查看详细的策略回测报告，包括总体表现、年度分析和月度统计
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：报告列表 */}
          <div className="space-y-4">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="w-full">
                <TabsTrigger value="total" className="flex-1">总报告</TabsTrigger>
                <TabsTrigger value="yearly" className="flex-1">年度</TabsTrigger>
                <TabsTrigger value="monthly" className="flex-1">月度</TabsTrigger>
              </TabsList>
            </Tabs>

            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  {typeIcons[activeTab]}
                  {typeLabels[activeTab]}
                </CardTitle>
                <CardDescription>
                  共 {filteredReports.length} 份报告
                </CardDescription>
              </CardHeader>
              <CardContent>
                {filteredReports.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    暂无{typeLabels[activeTab]}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {filteredReports.map((report) => (
                      <button
                        key={report.id}
                        onClick={() => setSelectedReport(report.path)}
                        className={`w-full text-left p-3 rounded-lg transition-colors flex items-center justify-between ${
                          selectedReport === report.path
                            ? "bg-primary/20 border border-primary/50"
                            : "bg-background/50 hover:bg-background/80"
                        }`}
                      >
                        <div>
                          <div className="font-medium">{report.name}</div>
                          <div className="text-xs text-muted-foreground">
                            更新于 {new Date(report.updatedAt).toLocaleDateString()}
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-muted-foreground" />
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 快速统计 */}
            <Card className="glass-card bg-gradient-to-br from-green-500/10 to-blue-500/10">
              <CardContent className="pt-6">
                <h3 className="font-semibold mb-4">📊 报告统计</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">总报告</span>
                    <span>{staticReports.filter((r) => r.type === "total").length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">年度报告</span>
                    <span>{staticReports.filter((r) => r.type === "yearly").length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">月度报告</span>
                    <span>{staticReports.filter((r) => r.type === "monthly").length}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 右侧：报告内容 */}
          <div className="lg:col-span-2">
            {selectedReportData ? (
              <Card className="glass-card">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">
                        {selectedReportData.name}
                      </CardTitle>
                      <CardDescription>
                        更新于 {new Date(selectedReportData.updatedAt).toLocaleString()}
                      </CardDescription>
                    </div>
                    <Badge variant="secondary">Markdown</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap text-sm bg-background/50 p-4 rounded-lg">
                      {selectedReportData.content}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="glass-card">
                <CardContent className="py-24 text-center text-muted-foreground">
                  <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg">选择左侧的报告查看详细内容</p>
                  <p className="text-sm mt-2">
                    报告包含策略的胜率、收益率、最优持有周期等详细数据
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
