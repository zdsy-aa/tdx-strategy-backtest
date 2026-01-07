import StrategyCard from "@/components/StrategyCard";
import strategiesData from "@/data/strategies.json";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Strategies() {
  const allStrategies = strategiesData.strategies;
  const steadyStrategies = allStrategies.filter(s => s.tag === "中长线" || s.tag === "全能");
  const aggressiveStrategies = allStrategies.filter(s => s.tag === "短线" || s.tag === "熊市");

  return (
    <div className="space-y-8">
      <div className="max-w-2xl">
        <h1 className="text-3xl font-bold mb-4">组合交易方案</h1>
        <p className="text-muted-foreground text-lg">
          单一指标往往存在局限性，通过多指标共振和相互验证，可以显著提高交易胜率。
          请根据您的风险承受能力和持仓周期选择合适的方案。
        </p>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList className="bg-white/5 border border-white/10 p-1 mb-8">
          <TabsTrigger value="all">全部方案</TabsTrigger>
          <TabsTrigger value="steady">稳健/中长线</TabsTrigger>
          <TabsTrigger value="aggressive">激进/短线</TabsTrigger>
        </TabsList>
        
        <TabsContent value="all" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allStrategies.map((strategy) => (
              <StrategyCard key={strategy.id} strategy={strategy} />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="steady" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {steadyStrategies.map((strategy) => (
              <StrategyCard key={strategy.id} strategy={strategy} />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="aggressive" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {aggressiveStrategies.map((strategy) => (
              <StrategyCard key={strategy.id} strategy={strategy} />
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
