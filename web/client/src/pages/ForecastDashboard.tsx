import { useState, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, Search, Calendar } from "lucide-react";
import Layout from "@/components/Layout";
import forecastSummaryRaw from "@/data/forecast_summary.json";

interface ForecastData {
  code: string;
  name: string;
  latest_close: number;
  kalman_price: number;
  particle_price: number;
  market_state: number;
  ensemble_forecast: number;
  forecast_change_pct: number;
  confidence: number;
  analysis_date: string;
  forecast_date: string;
  forecast_price?: number;
}

interface SummaryData {
  generated_at: string;
  total_stocks?: number;
  successful?: number;
  failed?: number;
  all_predictions?: ForecastData[];
  predictions?: ForecastData[];
}

export default function ForecastDashboard() {
  const summaryRaw = forecastSummaryRaw as unknown as SummaryData;
  const summaryData = useMemo(() => {
    return {
      generated_at: summaryRaw.generated_at,
      total_stocks: summaryRaw.total_stocks ?? (summaryRaw.predictions?.length || 0),
      successful: summaryRaw.successful ?? (summaryRaw.predictions?.length || 0),
      failed: summaryRaw.failed ?? 0,
      all_predictions: summaryRaw.all_predictions ?? summaryRaw.predictions ?? []
    };
  }, [summaryRaw]);
  
  const [searchTerm, setSearchTerm] = useState("");
  const [dateFilter, setDateFilter] = useState("all");
  const [selectedStock, setSelectedStock] = useState<ForecastData | null>(
    summaryData?.all_predictions?.[0] || null
  );

  const availableDates = useMemo(() => {
    if (!summaryData?.all_predictions) return [];
    const dates = Array.from(new Set(summaryData.all_predictions.map(p => p.analysis_date)));
    return dates.sort((a, b) => b.localeCompare(a));
  }, [summaryData]);

  const filteredData = useMemo(() => {
    if (!summaryData?.all_predictions) return [];
    return summaryData.all_predictions.filter(item => {
      const matchesSearch = item.code.includes(searchTerm) || item.name.includes(searchTerm);
      const matchesDate = dateFilter === "all" || item.analysis_date === dateFilter;
      return matchesSearch && matchesDate;
    });
  }, [summaryData, searchTerm, dateFilter]);

  if (!summaryData) {
    return (
      <Layout>
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="size-5 text-red-500" />
            <p className="text-red-500">无法加载预测数据。请确保已运行 a7_advanced_forecast.py 脚本。</p>
          </div>
        </div>
      </Layout>
    );
  }

  const getMarketStateLabel = (state: number) => {
    const labels = ["牛市", "熊市", "震荡"];
    return labels[state] || "未知";
  };

  const getMarketStateColor = (state: number) => {
    const colors = ["text-green-500", "text-red-500", "text-yellow-500"];
    return colors[state] || "text-gray-500";
  };

  return (
    <Layout>
      <div className="space-y-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold mb-2">📊 次日价格预测</h1>
            <p className="text-muted-foreground">
              基于多模型集成预测最新数据日期的次日表现 | 最后更新: {new Date(summaryData.generated_at).toLocaleString()}
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
              <Input 
                placeholder="搜索代码或名称..." 
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="flex items-center gap-2">
              <Calendar className="size-4 text-muted-foreground" />
              <Select value={dateFilter} onValueChange={setDateFilter}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="选择日期" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部日期</SelectItem>
                  {availableDates.map(date => (
                    <SelectItem key={date} value={date}>{date}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6">
          <Card className="glass-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <div>
                <CardTitle className="text-xl font-bold">预测排行榜</CardTitle>
                <CardDescription>按预测涨幅排序的股票列表 (当前筛选: {filteredData.length} 只)</CardDescription>
              </div>
              <Badge variant="outline" className="px-3 py-1">
                {selectedStock ? `已选中: ${selectedStock.name} (${selectedStock.code})` : "请选择股票"}
              </Badge>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent border-white/10">
                      <TableHead className="w-16">排名</TableHead>
                      <TableHead className="w-24">代码</TableHead>
                      <TableHead className="w-32">名称</TableHead>
                      <TableHead className="text-right">当前价格</TableHead>
                      <TableHead className="text-right">次日预测</TableHead>
                      <TableHead className="text-center">预测涨幅</TableHead>
                      <TableHead className="text-center">置信度</TableHead>
                      <TableHead className="text-center">市场状态</TableHead>
                      <TableHead className="text-right">预测日期</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredData.slice(0, 100).map((item, idx) => (
                      <TableRow 
                        key={idx} 
                        className={`cursor-pointer border-white/5 transition-all duration-200 ${
                          selectedStock?.code === item.code 
                            ? "outline-2 outline-primary outline-offset-[-2px] bg-primary/10 z-10" 
                            : "hover:bg-primary/5"
                        }`} 
                        onClick={() => setSelectedStock(item)}
                      >
                        <TableCell className="font-medium">{idx + 1}</TableCell>
                        <TableCell className="font-mono">{item.code}</TableCell>
                        <TableCell className="font-semibold">{item.name}</TableCell>
                        <TableCell className="text-right font-mono">¥{item.latest_close.toFixed(2)}</TableCell>
                        <TableCell className="text-right font-mono text-primary font-bold">¥{(item.ensemble_forecast ?? item.forecast_price ?? 0).toFixed(2)}</TableCell>
                        <TableCell className="text-center">
                          <Badge variant={item.forecast_change_pct > 0 ? "default" : "secondary"} className={item.forecast_change_pct > 0 ? "bg-green-500/80 hover:bg-green-500" : "bg-red-500/80 hover:bg-red-500"}>
                            {item.forecast_change_pct > 0 ? "+" : ""}
                            {item.forecast_change_pct.toFixed(2)}%
                          </Badge>
                        </TableCell>
                        <TableCell className="text-center">
                          <div className="flex items-center justify-center gap-2">
                            <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div className="h-full bg-primary" style={{ width: `${item.confidence * 100}%` }} />
                            </div>
                            <span className="text-xs font-mono">{(item.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </TableCell>
                        <TableCell className={`text-center font-medium ${getMarketStateColor(item.market_state)}`}>
                          {getMarketStateLabel(item.market_state)}
                        </TableCell>
                        <TableCell className="text-right text-muted-foreground text-xs font-mono">{item.forecast_date}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {filteredData.length > 100 && (
                  <p className="text-center text-sm text-muted-foreground mt-6 py-4 border-t border-white/5">仅显示前 100 条结果，请使用搜索或日期过滤缩小范围</p>
                )}
              </div>
            </CardContent>
          </Card>

          {selectedStock && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">模型分析: {selectedStock.name}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                      <p className="text-xs text-muted-foreground mb-1">卡尔曼滤波平滑价</p>
                      <p className="text-xl font-bold">¥{(selectedStock.kalman_price ?? selectedStock.latest_close).toFixed(2)}</p>
                    </div>
                    <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                      <p className="text-xs text-muted-foreground mb-1">粒子滤波预测价</p>
                      <p className="text-xl font-bold">¥{(selectedStock.particle_price ?? selectedStock.latest_close).toFixed(2)}</p>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-semibold text-primary">集成模型预测结论</span>
                      <Badge className="bg-primary text-white">{(selectedStock.confidence * 100).toFixed(0)}% 置信度</Badge>
                    </div>
                    <p className="text-sm">
                      基于多种模型综合研判，预计该股在 <span className="font-bold underline">{selectedStock.forecast_date}</span> 的收盘价约为 
                      <span className="text-lg font-bold mx-1 text-primary">¥{(selectedStock.ensemble_forecast ?? selectedStock.forecast_price ?? 0).toFixed(2)}</span>，
                      较当前价格变动幅度约为 <span className={`font-bold ${selectedStock.forecast_change_pct >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {selectedStock.forecast_change_pct > 0 ? '+' : ''}{selectedStock.forecast_change_pct.toFixed(2)}%
                      </span>。
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">市场状态: {getMarketStateLabel(selectedStock.market_state)}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center justify-center h-full py-4">
                    <div className={`text-4xl font-bold mb-2 ${getMarketStateColor(selectedStock.market_state)}`}>
                      {selectedStock.market_state === 0 ? "🐂 BULLISH" : selectedStock.market_state === 1 ? "🐻 BEARISH" : "⚖️ NEUTRAL"}
                    </div>
                    <p className="text-sm text-center text-muted-foreground">
                      隐马尔可夫模型 (HMM) 识别当前市场处于{getMarketStateLabel(selectedStock.market_state)}阶段。
                      建议：{selectedStock.market_state === 0 ? "积极关注，顺势而为" : selectedStock.market_state === 1 ? "谨慎观望，注意风险" : "高抛低吸，震荡操作"}。
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
