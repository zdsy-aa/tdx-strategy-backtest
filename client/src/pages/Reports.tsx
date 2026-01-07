import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, FileText, Calendar, TrendingUp, ChevronRight } from "lucide-react";
import { Streamdown } from "streamdown";
import { trpc } from "@/lib/trpc";

export default function Reports() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("total");

  const { data: reports, isLoading: reportsLoading } = trpc.reports.list.useQuery({});
  
  const { data: reportContent, isLoading: contentLoading } = trpc.reports.get.useQuery(
    { path: selectedReport! },
    { enabled: !!selectedReport }
  );

  const filteredReports = reports?.filter((r) => r.type === activeTab) || [];

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
                {reportsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin" />
                  </div>
                ) : filteredReports.length === 0 ? (
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
                    <span>{reports?.filter((r) => r.type === "total").length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">年度报告</span>
                    <span>{reports?.filter((r) => r.type === "yearly").length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">月度报告</span>
                    <span>{reports?.filter((r) => r.type === "monthly").length || 0}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 右侧：报告内容 */}
          <div className="lg:col-span-2">
            {selectedReport ? (
              <Card className="glass-card">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">
                        {reports?.find((r) => r.path === selectedReport)?.name}
                      </CardTitle>
                      <CardDescription>
                        {reportContent && (
                          <>更新于 {new Date(reportContent.updatedAt).toLocaleString()}</>
                        )}
                      </CardDescription>
                    </div>
                    <Badge variant="secondary">
                      {reportContent?.isMarkdown ? "Markdown" : "CSV"}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {contentLoading ? (
                    <div className="flex items-center justify-center py-16">
                      <Loader2 className="w-8 h-8 animate-spin" />
                    </div>
                  ) : reportContent?.isMarkdown ? (
                    <div className="prose prose-invert max-w-none">
                      <Streamdown>{reportContent.content}</Streamdown>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <pre className="text-sm bg-background/50 p-4 rounded-lg overflow-x-auto">
                        {reportContent?.content}
                      </pre>
                    </div>
                  )}
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
