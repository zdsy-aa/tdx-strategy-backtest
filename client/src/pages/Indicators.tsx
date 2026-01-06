import IndicatorCard from "@/components/IndicatorCard";
import strategiesData from "@/data/strategies.json";

export default function Indicators() {
  return (
    <div className="space-y-8">
      <div className="max-w-2xl">
        <h1 className="text-3xl font-bold mb-4">核心指标详解</h1>
        <p className="text-muted-foreground text-lg">
          深入理解每个指标的计算逻辑、买卖信号及其背后的市场含义。
          掌握这些基础是构建成功交易系统的关键。
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {strategiesData.indicators.map((indicator) => (
          <IndicatorCard key={indicator.id} indicator={indicator} />
        ))}
      </div>
    </div>
  );
}
