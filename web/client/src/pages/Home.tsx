import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { ArrowRight, BarChart2, ShieldCheck, Zap } from "lucide-react";
import { Link } from "wouter";
import strategiesData from "@/data/strategies.json";
import StrategyCard from "@/components/StrategyCard";
import Layout from "@/components/Layout";

export default function Home() {
  const { user, loading, isAuthenticated } = useAuth();
  const featuredStrategies = strategiesData.strategies.slice(0, 3);

  return (
    <Layout>
      <div className="space-y-12">
        {/* Hero Section */}
        <section className="relative py-20 overflow-hidden rounded-3xl border border-white/10 bg-black/40 backdrop-blur-sm">
          <div className="absolute inset-0 bg-[url('/images/hero-bg.png')] bg-cover bg-center opacity-30" />
          <div className="absolute inset-0 bg-gradient-to-r from-background via-background/80 to-transparent" />
          
          <div className="container relative z-10">
            <div className="max-w-2xl space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
                v2.0 策略系统已上线
              </div>
              
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white">
                通达信指标 <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-purple-400">
                  实战交易参考系统
                </span>
              </h1>
              
              <p className="text-lg text-muted-foreground leading-relaxed">
                基于六脉神剑、买卖点、摇钱树等核心指标构建的量化交易策略体系。
                提供稳健、激进、共振等多种组合方案，助您在不同市场环境下做出更优决策。
              </p>
              
              <div className="flex flex-wrap gap-4 pt-4">
                <Link href="/strategies">
                  <Button size="lg" className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/25">
                    浏览组合方案 <ArrowRight className="ml-2 size-4" />
                  </Button>
                </Link>
                <Link href="/indicators">
                  <Button size="lg" variant="outline" className="bg-white/5 border-white/10 hover:bg-white/10 text-white">
                    查看指标详解
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
            <div className="size-12 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4 text-blue-400">
              <ShieldCheck className="size-6" />
            </div>
            <h3 className="text-xl font-bold mb-2">稳健风控</h3>
            <p className="text-muted-foreground">
              基于缠论结构和多指标共振，严格控制回撤，适合中长线资金配置。
            </p>
          </div>
          
          <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
            <div className="size-12 rounded-xl bg-purple-500/20 flex items-center justify-center mb-4 text-purple-400">
              <Zap className="size-6" />
            </div>
            <h3 className="text-xl font-bold mb-2">激进博弈</h3>
            <p className="text-muted-foreground">
              利用买点2和六脉神剑爆发点，捕捉短线主升浪，追求资金利用率。
            </p>
          </div>
          
          <div className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
            <div className="size-12 rounded-xl bg-green-500/20 flex items-center justify-center mb-4 text-green-400">
              <BarChart2 className="size-6" />
            </div>
            <h3 className="text-xl font-bold mb-2">量化回测</h3>
            <p className="text-muted-foreground">
              所有策略均经过10年历史数据回测验证，提供真实的胜率和收益参考。
            </p>
          </div>
        </section>

        {/* Featured Strategies */}
        <section>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold mb-2">精选策略组合</h2>
              <p className="text-muted-foreground">根据不同风险偏好推荐的最优方案</p>
            </div>
            <Link href="/strategies">
              <Button variant="ghost" className="text-primary hover:text-primary/80">
                查看全部 <ArrowRight className="ml-2 size-4" />
              </Button>
            </Link>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {featuredStrategies.map((strategy) => (
              <StrategyCard key={strategy.id} strategy={strategy} />
            ))}
          </div>
        </section>
      </div>
    </Layout>
  );
}
