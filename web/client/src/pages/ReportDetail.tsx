import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState, useMemo } from "react";
import { Search, Download, ArrowUpDown } from "lucide-react";
import Layout from "@/components/Layout";
import stockReportsData from "@/data/stock_reports.json";
import backtestData from "@/data/backtest_results.json";

interface StockReport {
  code: string;
  name: string;
  totalReturn: string;
  yearReturn: string;
  monthReturn: string;
  totalWinRate: string;
  yearWinRate: string;
  monthWinRate: string;
  totalTrades: number;
  yearTrades: number;
  monthTrades: number;
  lastSignal: string;
  lastSignalDate: string;
}

// 默认示例数据，当 stock_reports.json 为空时使用
const defaultReports: StockReport[] = [
  { code: "600519", name: "贵州茅台", totalReturn: "55.8%", yearReturn: "18.5%", monthReturn: "5.2%", totalWinRate: "78.5%", yearWinRate: "82.0%", monthWinRate: "88.0%", totalTrades: 65, yearTrades: 14, monthTrades: 7, lastSignal: "六脉6红", lastSignalDate: "2025-01-06" },
  { code: "000858", name: "五粮液", totalReturn: "42.5%", yearReturn: "15.8%", monthReturn: "4.2%", totalWinRate: "72.5%", yearWinRate: "78.0%", monthWinRate: "85.0%", totalTrades: 58, yearTrades: 12, monthTrades: 6, lastSignal: "买点2", lastSignalDate: "2025-01-04" },
  { code: "000333", name: "美的集团", totalReturn: "35.2%", yearReturn: "12.5%", monthReturn: "3.5%", totalWinRate: "68.5%", yearWinRate: "72.0%", monthWinRate: "80.0%", totalTrades: 54, yearTrades: 9, monthTrades: 5, lastSignal: "六脉6红", lastSignalDate: "2025-01-06" },
  { code: "000063", name: "中兴通讯", totalReturn: "28.5%", yearReturn: "8.5%", monthReturn: "2.8%", totalWinRate: "65.0%", yearWinRate: "70.0%", monthWinRate: "75.0%", totalTrades: 60, yearTrades: 10, monthTrades: 4, lastSignal: "买点2", lastSignalDate: "2025-01-05" },
  { code: "600036", name: "招商银行", totalReturn: "22.5%", yearReturn: "7.8%", monthReturn: "2.5%", totalWinRate: "62.5%", yearWinRate: "68.0%", monthWinRate: "72.0%", totalTrades: 48, yearTrades: 8, monthTrades: 4, lastSignal: "六脉5红", lastSignalDate: "2025-01-02" },
  { code: "600887", name: "伊利股份", totalReturn: "25.2%", yearReturn: "8.2%", monthReturn: "2.8%", totalWinRate: "60.0%", yearWinRate: "65.0%", monthWinRate: "70.0%", totalTrades: 50, yearTrades: 8, monthTrades: 4, lastSignal: "买点2", lastSignalDate: "2025-01-05" },
  { code: "000001", name: "平安银行", totalReturn: "12.5%", yearReturn: "3.2%", monthReturn: "1.5%", totalWinRate: "58.3%", yearWinRate: "62.5%", monthWinRate: "66.7%", totalTrades: 48, yearTrades: 8, monthTrades: 3, lastSignal: "六脉5红", lastSignalDate: "2025-01-03" },
  { code: "601318", name: "中国平安", totalReturn: "15.5%", yearReturn: "4.5%", monthReturn: "1.5%", totalWinRate: "56.0%", yearWinRate: "60.0%", monthWinRate: "62.0%", totalTrades: 45, yearTrades: 7, monthTrades: 3, lastSignal: "无", lastSignalDate: "-" },
  { code: "000651", name: "格力电器", totalReturn: "18.5%", yearReturn: "5.2%", monthReturn: "1.2%", totalWinRate: "55.0%", yearWinRate: "58.0%", monthWinRate: "60.0%", totalTrades: 40, yearTrades: 6, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
  { code: "601398", name: "工商银行", totalReturn: "10.2%", yearReturn: "3.0%", monthReturn: "1.0%", totalWinRate: "54.0%", yearWinRate: "56.0%", monthWinRate: "58.0%", totalTrades: 38, yearTrades: 5, monthTrades: 2, lastSignal: "六脉4红", lastSignalDate: "2025-01-03" },
  { code: "600000", name: "浦发银行", totalReturn: "8.2%", yearReturn: "2.5%", monthReturn: "0.8%", totalWinRate: "52.0%", yearWinRate: "55.0%", monthWinRate: "50.0%", totalTrades: 35, yearTrades: 4, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
  { code: "000002", name: "万科A", totalReturn: "-5.2%", yearReturn: "-2.1%", monthReturn: "-0.8%", totalWinRate: "45.2%", yearWinRate: "42.0%", monthWinRate: "40.0%", totalTrades: 42, yearTrades: 5, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
];

export default function ReportDetail() {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<keyof StockReport>("totalWinRate");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [marketFilter, setMarketFilter] = useState<"all" | "sh" | "sz" | "bj">("all");

  // 根据股票代码识别市场（备用，优先使用数据中的 market 字段）
  const getMarketFromCode = (code: string): "sh" | "sz" | "bj" => {
    if (code.startsWith('6')) return 'sh';  // 沪市
    if (code.startsWith('8') || code.startsWith('4')) return 'bj';  // 北交所
    return 'sz';  // 深市
  };

  // 获取数据更新日期
  const lastUpdateDate = useMemo(() => {
    const data = backtestData as any;
    return data.last_update || new Date().toISOString().split('T')[0];
  }, []);

  const stockReports: StockReport[] = useMemo(() => {
    const data = stockReportsData as StockReport[];
    return data.length > 0 ? data : defaultReports;
  }, []);

  const filteredReports = stockReports
    .filter(report => {
      // 搜索筛选
      const matchSearch = report.code.includes(searchTerm) || 
        report.name.toLowerCase().includes(searchTerm.toLowerCase());
      
      // 市场筛选（优先使用数据中的 market 字段，否则根据代码推断）
      const reportMarket = (report as any).market || getMarketFromCode(report.code);
      const matchMarket = marketFilter === "all" || reportMarket === marketFilter;
      
      return matchSearch && matchMarket;
    })
    .sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      
      if (typeof aValue === "string" && typeof bValue === "string") {
        const aNum = parseFloat(aValue.replace("%", ""));
        const bNum = parseFloat(bValue.replace("%", ""));
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortOrder === "asc" ? aNum - bNum : bNum - aNum;
        }
        return sortOrder === "asc" 
          ? aValue.localeCompare(bValue) 
          : bValue.localeCompare(aValue);
      }
      
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortOrder === "asc" ? aValue - bValue : bValue - aValue;
      }
      
      return 0;
    });

  const handleSort = (field: keyof StockReport) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  const exportToCSV = () => {
    const headers = ["股票代码", "股票名称", "总收益", "年收益", "月收益", "总胜率", "年胜率", "月胜率", "总交易次数", "年交易次数", "月交易次数", "最新信号", "信号日期"];
    const rows = filteredReports.map(r => [
      r.code, r.name, r.totalReturn, r.yearReturn, r.monthReturn,
      r.totalWinRate, r.yearWinRate, r.monthWinRate,
      r.totalTrades, r.yearTrades, r.monthTrades,
      r.lastSignal, r.lastSignalDate
    ]);
    
    const csvContent = [headers, ...rows].map(row => row.join(",")).join("\n");
    const blob = new Blob(["\ufeff" + csvContent], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `stock_report_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
  };

  // 分页逻辑
  const totalPages = Math.ceil(filteredReports.length / pageSize);
  const paginatedReports = filteredReports.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  // 当搜索条件或市场筛选变化时重置到第一页
  useMemo(() => {
    setCurrentPage(1);
  }, [searchTerm, sortField, sortOrder, marketFilter]);

  const summary = useMemo(() => {
    if (filteredReports.length === 0) {
      return { avgWinRate: "0.0%", avgReturn: "0.0%", signalCount: 0, totalCount: 0 };
    }
    const totalWinRate = filteredReports.reduce((acc, r) => acc + parseFloat(r.totalWinRate), 0) / filteredReports.length;
    const totalReturn = filteredReports.reduce((acc, r) => acc + parseFloat(r.totalReturn), 0) / filteredReports.length;
    const signalCount = filteredReports.filter(r => r.lastSignal && r.lastSignal !== "无").length;
    return {
      avgWinRate: totalWinRate.toFixed(1) + "%",
      avgReturn: totalReturn.toFixed(1) + "%",
      signalCount,
      totalCount: filteredReports.length,
    };
  }, [filteredReports]);

  return (
    <Layout>
      <div className="container py-8 space-y-6">
        <div>
          <h1 className="text-3xl font-bold">报告明细</h1>
          <p className="text-muted-foreground">截止到 {lastUpdateDate}，所有股票的回测统计数据。包含总收益、年收益、月收益以及各维度的胜率统计。</p>
        </div>

        <Card className="glass-card">
          <CardHeader>
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <CardTitle>股票回测明细表</CardTitle>
                <p className="text-sm text-muted-foreground">类似Excel的数据展示，支持搜索、排序和导出</p>
              </div>
              <div className="flex gap-2 w-full md:w-auto flex-wrap">
                <div className="relative w-full md:w-64">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground size-4" />
                  <Input 
                    placeholder="搜索股票代码或名称..." 
                    className="pl-10" 
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
                <Select value={marketFilter} onValueChange={(value) => setMarketFilter(value as "all" | "sh" | "sz" | "bj")}>
                  <SelectTrigger className="w-[120px]">
                    <SelectValue placeholder="市场筛选" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">全部市场</SelectItem>
                    <SelectItem value="sh">沪市 (6开头)</SelectItem>
                    <SelectItem value="sz">深市 (0/3开头)</SelectItem>
                    <SelectItem value="bj">北交所 (4/8开头)</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={sortField} onValueChange={(value) => handleSort(value as keyof StockReport)}>
                  <SelectTrigger className="w-[120px]">
                    <SelectValue placeholder="排序方式" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="totalWinRate">总胜率</SelectItem>
                    <SelectItem value="totalReturn">总收益</SelectItem>
                    <SelectItem value="yearWinRate">年胜率</SelectItem>
                    <SelectItem value="yearReturn">年收益</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline" onClick={exportToCSV}><Download className="mr-2 size-4" /> 导出CSV</Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-white/5 border-white/10">
                    <TableHead>股票代码</TableHead>
                    <TableHead>股票名称</TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("totalReturn")} className="px-2 py-1 h-auto">总收益<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("yearReturn")} className="px-2 py-1 h-auto">年收益<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("monthReturn")} className="px-2 py-1 h-auto">月收益<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("totalWinRate")} className="px-2 py-1 h-auto">总胜率<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("yearWinRate")} className="px-2 py-1 h-auto">年胜率<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead><Button variant="ghost" onClick={() => handleSort("monthWinRate")} className="px-2 py-1 h-auto">月胜率<ArrowUpDown className="ml-2 size-3" /></Button></TableHead>
                    <TableHead>总交易</TableHead>
                    <TableHead>年交易</TableHead>
                    <TableHead>月交易</TableHead>
                    <TableHead>最新信号</TableHead>
                    <TableHead>信号日期</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedReports.map((report) => (
                    <TableRow key={report.code} className="hover:bg-white/5 border-white/10">
                      <TableCell className="font-mono">{report.code}</TableCell>
                      <TableCell className="font-medium">{report.name}</TableCell>
                      <TableCell className={parseFloat(report.totalReturn) >= 0 ? "text-red-400 font-semibold" : "text-green-500"}>{report.totalReturn}</TableCell>
                      <TableCell className={parseFloat(report.yearReturn) >= 0 ? "text-red-400" : "text-green-500"}>{report.yearReturn}</TableCell>
                      <TableCell className={parseFloat(report.monthReturn) >= 0 ? "text-red-400" : "text-green-500"}>{report.monthReturn}</TableCell>
                      <TableCell className="text-yellow-400 font-bold">{report.totalWinRate}</TableCell>
                      <TableCell>{report.yearWinRate}</TableCell>
                      <TableCell>{report.monthWinRate}</TableCell>
                      <TableCell>{report.totalTrades}</TableCell>
                      <TableCell>{report.yearTrades}</TableCell>
                      <TableCell>{report.monthTrades}</TableCell>
                      <TableCell>
                        <Badge variant={report.lastSignal === "无" ? "secondary" : "default"}>{report.lastSignal}</Badge>
                      </TableCell>
                      <TableCell>{report.lastSignalDate}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            
            {/* 分页控件 */}
            <div className="flex items-center justify-between px-2 py-4 border-t border-white/10">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  显示 {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, filteredReports.length)} 条，共 {filteredReports.length} 条
                </span>
                <Select value={pageSize.toString()} onValueChange={(value) => { setPageSize(Number(value)); setCurrentPage(1); }}>
                  <SelectTrigger className="w-[100px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="20">20 条/页</SelectItem>
                    <SelectItem value="50">50 条/页</SelectItem>
                    <SelectItem value="100">100 条/页</SelectItem>
                    <SelectItem value="200">200 条/页</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setCurrentPage(1)} 
                  disabled={currentPage === 1}
                >
                  首页
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setCurrentPage(currentPage - 1)} 
                  disabled={currentPage === 1}
                >
                  上一页
                </Button>
                <span className="text-sm text-muted-foreground px-4">
                  第 {currentPage} / {totalPages} 页
                </span>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setCurrentPage(currentPage + 1)} 
                  disabled={currentPage === totalPages}
                >
                  下一页
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setCurrentPage(totalPages)} 
                  disabled={currentPage === totalPages}
                >
                  尾页
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="glass-card">
            <CardContent className="pt-6 text-center">
              <div className="text-sm text-muted-foreground">平均总胜率</div>
              <div className="text-2xl font-bold text-green-400">{summary.avgWinRate}</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="pt-6 text-center">
              <div className="text-sm text-muted-foreground">平均总收益</div>
              <div className="text-2xl font-bold">{summary.avgReturn}</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="pt-6 text-center">
              <div className="text-sm text-muted-foreground">有信号股票数</div>
              <div className="text-2xl font-bold">{summary.signalCount}</div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="pt-6 text-center">
              <div className="text-sm text-muted-foreground">总股票数</div>
              <div className="text-2xl font-bold">{summary.totalCount}</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
