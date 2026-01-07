import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState } from "react";
import { Search, Download, ArrowUpDown } from "lucide-react";
import Layout from "@/components/Layout";

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

export default function ReportDetail() {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<keyof StockReport>("totalWinRate");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // 模拟股票报告数据（截止到20250106）
  const stockReports: StockReport[] = [
    { code: "000001", name: "平安银行", totalReturn: "12.5%", yearReturn: "3.2%", monthReturn: "1.5%", totalWinRate: "58.3%", yearWinRate: "62.5%", monthWinRate: "66.7%", totalTrades: 48, yearTrades: 8, monthTrades: 3, lastSignal: "六脉5红", lastSignalDate: "2025-01-03" },
    { code: "000002", name: "万科A", totalReturn: "-5.2%", yearReturn: "-2.1%", monthReturn: "-0.8%", totalWinRate: "45.2%", yearWinRate: "42.0%", monthWinRate: "40.0%", totalTrades: 42, yearTrades: 5, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
    { code: "000063", name: "中兴通讯", totalReturn: "28.5%", yearReturn: "8.5%", monthReturn: "2.8%", totalWinRate: "65.0%", yearWinRate: "70.0%", monthWinRate: "75.0%", totalTrades: 60, yearTrades: 10, monthTrades: 4, lastSignal: "买点2", lastSignalDate: "2025-01-05" },
    { code: "000333", name: "美的集团", totalReturn: "35.2%", yearReturn: "12.5%", monthReturn: "3.5%", totalWinRate: "68.5%", yearWinRate: "72.0%", monthWinRate: "80.0%", totalTrades: 54, yearTrades: 9, monthTrades: 5, lastSignal: "六脉6红", lastSignalDate: "2025-01-06" },
    { code: "000651", name: "格力电器", totalReturn: "18.5%", yearReturn: "5.2%", monthReturn: "1.2%", totalWinRate: "55.0%", yearWinRate: "58.0%", monthWinRate: "60.0%", totalTrades: 40, yearTrades: 6, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
    { code: "000858", name: "五粮液", totalReturn: "42.5%", yearReturn: "15.8%", monthReturn: "4.2%", totalWinRate: "72.5%", yearWinRate: "78.0%", monthWinRate: "85.0%", totalTrades: 58, yearTrades: 12, monthTrades: 6, lastSignal: "买点2", lastSignalDate: "2025-01-04" },
    { code: "600000", name: "浦发银行", totalReturn: "8.2%", yearReturn: "2.5%", monthReturn: "0.8%", totalWinRate: "52.0%", yearWinRate: "55.0%", monthWinRate: "50.0%", totalTrades: 35, yearTrades: 4, monthTrades: 2, lastSignal: "无", lastSignalDate: "-" },
    { code: "600036", name: "招商银行", totalReturn: "22.5%", yearReturn: "7.8%", monthReturn: "2.5%", totalWinRate: "62.5%", yearWinRate: "68.0%", monthWinRate: "72.0%", totalTrades: 48, yearTrades: 8, monthTrades: 4, lastSignal: "六脉5红", lastSignalDate: "2025-01-02" },
    { code: "600519", name: "贵州茅台", totalReturn: "55.8%", yearReturn: "18.5%", monthReturn: "5.2%", totalWinRate: "78.5%", yearWinRate: "82.0%", monthWinRate: "88.0%", totalTrades: 65, yearTrades: 14, monthTrades: 7, lastSignal: "六脉6红", lastSignalDate: "2025-01-06" },
    { code: "600887", name: "伊利股份", totalReturn: "25.2%", yearReturn: "8.2%", monthReturn: "2.8%", totalWinRate: "60.0%", yearWinRate: "65.0%", monthWinRate: "70.0%", totalTrades: 50, yearTrades: 8, monthTrades: 4, lastSignal: "买点2", lastSignalDate: "2025-01-05" },
    { code: "601318", name: "中国平安", totalReturn: "15.5%", yearReturn: "4.5%", monthReturn: "1.5%", totalWinRate: "56.0%", yearWinRate: "60.0%", monthWinRate: "62.0%", totalTrades: 45, yearTrades: 7, monthTrades: 3, lastSignal: "无", lastSignalDate: "-" },
    { code: "601398", name: "工商银行", totalReturn: "10.2%", yearReturn: "3.0%", monthReturn: "1.0%", totalWinRate: "54.0%", yearWinRate: "56.0%", monthWinRate: "58.0%", totalTrades: 38, yearTrades: 5, monthTrades: 2, lastSignal: "六脉4红", lastSignalDate: "2025-01-03" },
  ];

  // 过滤和排序
  const filteredReports = stockReports
    .filter(report => 
      report.code.includes(searchTerm) || 
      report.name.includes(searchTerm)
    )
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

  return (
    <Layout>
      <div className="space-y-8">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-bold mb-4">报告明细</h1>
          <p className="text-muted-foreground text-lg">
            截止到2025年1月6日，所有股票的回测统计数据。包含总收益、年收益、月收益以及各维度的胜率统计。
          </p>
        </div>

        <Card className="glass-card">
          <CardHeader>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <CardTitle>股票回测明细表</CardTitle>
                <CardDescription>类似Excel的数据展示，支持搜索、排序和导出</CardDescription>
              </div>
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
                  <Input
                    placeholder="搜索股票代码或名称..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 w-[200px] bg-white/5 border-white/10"
                  />
                </div>
                <Select value={sortField} onValueChange={(v) => setSortField(v as keyof StockReport)}>
                  <SelectTrigger className="w-[150px] bg-white/5 border-white/10">
                    <SelectValue placeholder="排序字段" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="totalWinRate">总胜率</SelectItem>
                    <SelectItem value="yearWinRate">年胜率</SelectItem>
                    <SelectItem value="monthWinRate">月胜率</SelectItem>
                    <SelectItem value="totalReturn">总收益</SelectItem>
                    <SelectItem value="yearReturn">年收益</SelectItem>
                    <SelectItem value="monthReturn">月收益</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline" onClick={exportToCSV} className="bg-white/5 border-white/10">
                  <Download className="size-4 mr-2" />
                  导出CSV
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-white/5 border-white/10">
                    <TableHead className="sticky left-0 bg-background z-10">股票代码</TableHead>
                    <TableHead className="sticky left-[100px] bg-background z-10">股票名称</TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("totalReturn")}>
                      <div className="flex items-center justify-end gap-1">
                        总收益
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("yearReturn")}>
                      <div className="flex items-center justify-end gap-1">
                        年收益
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("monthReturn")}>
                      <div className="flex items-center justify-end gap-1">
                        月收益
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("totalWinRate")}>
                      <div className="flex items-center justify-end gap-1">
                        总胜率
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("yearWinRate")}>
                      <div className="flex items-center justify-end gap-1">
                        年胜率
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => handleSort("monthWinRate")}>
                      <div className="flex items-center justify-end gap-1">
                        月胜率
                        <ArrowUpDown className="size-3" />
                      </div>
                    </TableHead>
                    <TableHead className="text-right">总交易</TableHead>
                    <TableHead className="text-right">年交易</TableHead>
                    <TableHead className="text-right">月交易</TableHead>
                    <TableHead>最新信号</TableHead>
                    <TableHead>信号日期</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredReports.map((report) => {
                    const totalReturnNum = parseFloat(report.totalReturn);
                    const yearReturnNum = parseFloat(report.yearReturn);
                    const monthReturnNum = parseFloat(report.monthReturn);
                    const totalWinRateNum = parseFloat(report.totalWinRate);
                    
                    return (
                      <TableRow key={report.code} className="hover:bg-white/5 border-white/10">
                        <TableCell className="font-mono sticky left-0 bg-background z-10">{report.code}</TableCell>
                        <TableCell className="font-medium sticky left-[100px] bg-background z-10">{report.name}</TableCell>
                        <TableCell className={`text-right font-bold ${totalReturnNum >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {report.totalReturn}
                        </TableCell>
                        <TableCell className={`text-right ${yearReturnNum >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {report.yearReturn}
                        </TableCell>
                        <TableCell className={`text-right ${monthReturnNum >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {report.monthReturn}
                        </TableCell>
                        <TableCell className={`text-right font-bold ${
                          totalWinRateNum >= 60 ? 'text-green-400' : 
                          totalWinRateNum >= 50 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {report.totalWinRate}
                        </TableCell>
                        <TableCell className="text-right">{report.yearWinRate}</TableCell>
                        <TableCell className="text-right">{report.monthWinRate}</TableCell>
                        <TableCell className="text-right text-muted-foreground">{report.totalTrades}</TableCell>
                        <TableCell className="text-right text-muted-foreground">{report.yearTrades}</TableCell>
                        <TableCell className="text-right text-muted-foreground">{report.monthTrades}</TableCell>
                        <TableCell>
                          {report.lastSignal !== "无" ? (
                            <Badge className="bg-primary/20 text-primary hover:bg-primary/30">
                              {report.lastSignal}
                            </Badge>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell className="text-muted-foreground">{report.lastSignalDate}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* 统计摘要 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">平均总胜率</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-400">
                {(stockReports.reduce((sum, r) => sum + parseFloat(r.totalWinRate), 0) / stockReports.length).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">平均总收益</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-400">
                {(stockReports.reduce((sum, r) => sum + parseFloat(r.totalReturn), 0) / stockReports.length).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">有信号股票数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-primary">
                {stockReports.filter(r => r.lastSignal !== "无").length}
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">总股票数</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {stockReports.length}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
