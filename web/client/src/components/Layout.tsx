import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Activity, BarChart2, BookOpen, Layers, Brain, Settings2, FileText, Menu, X, Table2, TrendingUp, BarChart3 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";

export default function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const mainNavItems = [
    { href: "/", label: "仪表盘", icon: Activity },
    { href: "/indicators", label: "指标详解", icon: BookOpen },
    { href: "/strategies", label: "指标方案", icon: Layers },
    { href: "/backtest", label: "回测数据", icon: BarChart2 },
    { href: "/report-detail", label: "报告明细", icon: Table2 },
    { href: "/visual-buy-points", label: "可视化买点", icon: TrendingUp },
    { href: "/analysis-report", label: "分析报告", icon: BarChart3 },
  ];

  const toolNavItems = [
    { href: "/ai-optimizer", label: "AI 策略优化", icon: Brain },
    { href: "/custom-strategy", label: "自定义策略", icon: Settings2 },
    { href: "/reports", label: "回测报告", icon: FileText },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground font-sans selection:bg-primary/30">
      {/* Navbar */}
      <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-background/80 backdrop-blur-xl">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded-lg bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg shadow-primary/20">
              <Activity className="size-5 text-white" />
            </div>
            <span className="text-lg font-bold tracking-tight">
              Trade<span className="text-primary">Guide</span>
            </span>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-6">
            {mainNavItems.map((item) => (
              <Link key={item.href} href={item.href}>
                <span
                  className={cn(
                    "flex items-center gap-2 text-sm font-medium transition-colors hover:text-primary cursor-pointer",
                    location === item.href
                      ? "text-primary"
                      : "text-muted-foreground"
                  )}
                >
                  <item.icon className="size-4" />
                  {item.label}
                </span>
              </Link>
            ))}
            
            {/* Tools Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-primary">
                  <Settings2 className="size-4 mr-2" />
                  工具
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {toolNavItems.map((item) => (
                  <DropdownMenuItem key={item.href} asChild>
                    <Link href={item.href}>
                      <span className="flex items-center gap-2 w-full cursor-pointer">
                        <item.icon className="size-4" />
                        {item.label}
                      </span>
                    </Link>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </nav>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </Button>
          
          <div className="hidden sm:flex items-center gap-4">
            <div className="text-xs text-muted-foreground">
              <span className="inline-block size-2 rounded-full bg-green-500 mr-2 animate-pulse"></span>
              系统运行中
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="lg:hidden border-t border-white/10 bg-background/95 backdrop-blur-xl">
            <nav className="container py-4 space-y-2">
              {mainNavItems.map((item) => (
                <Link key={item.href} href={item.href}>
                  <span
                    onClick={() => setMobileMenuOpen(false)}
                    className={cn(
                      "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer",
                      location === item.href
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-white/5"
                    )}
                  >
                    <item.icon className="size-4" />
                    {item.label}
                  </span>
                </Link>
              ))}
              <div className="border-t border-white/10 pt-2 mt-2">
                <p className="px-3 py-1 text-xs text-muted-foreground">工具</p>
                {toolNavItems.map((item) => (
                  <Link key={item.href} href={item.href}>
                    <span
                      onClick={() => setMobileMenuOpen(false)}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer",
                        location === item.href
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-white/5"
                      )}
                    >
                      <item.icon className="size-4" />
                      {item.label}
                    </span>
                  </Link>
                ))}
              </div>
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 container py-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black/20 py-8 mt-auto">
        <div className="container flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            © 2026 TradeGuide. 仅供学习研究，不构成投资建议。
          </p>
          <div className="flex gap-4 text-sm text-muted-foreground">
            <a href="#" className="hover:text-primary transition-colors">免责声明</a>
            <a href="#" className="hover:text-primary transition-colors">关于我们</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
