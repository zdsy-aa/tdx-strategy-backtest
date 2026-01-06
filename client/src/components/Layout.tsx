import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Activity, BarChart2, BookOpen, Layers } from "lucide-react";

export default function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  const navItems = [
    { href: "/", label: "仪表盘", icon: Activity },
    { href: "/indicators", label: "指标详解", icon: BookOpen },
    { href: "/strategies", label: "组合方案", icon: Layers },
    { href: "/backtest", label: "回测数据", icon: BarChart2 },
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

          <nav className="hidden md:flex items-center gap-6">
            {navItems.map((item) => (
              <Link key={item.href} href={item.href}>
                <a
                  className={cn(
                    "flex items-center gap-2 text-sm font-medium transition-colors hover:text-primary",
                    location === item.href
                      ? "text-primary"
                      : "text-muted-foreground"
                  )}
                >
                  <item.icon className="size-4" />
                  {item.label}
                </a>
              </Link>
            ))}
          </nav>
          
          <div className="flex items-center gap-4">
            <div className="text-xs text-muted-foreground hidden sm:block">
              <span className="inline-block size-2 rounded-full bg-green-500 mr-2 animate-pulse"></span>
              系统运行中
            </div>
          </div>
        </div>
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
