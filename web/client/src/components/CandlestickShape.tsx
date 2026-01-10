import React from 'react';

interface CandlestickShapeProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: any;
  // Recharts会传递这些额外的props
  [key: string]: any;
}

/**
 * 自定义蜡烛图形状组件
 * 红色：收盘价 > 开盘价（上涨）
 * 绿色：收盘价 < 开盘价（下跌）
 */
export const CandlestickShape: React.FC<CandlestickShapeProps> = (props) => {
  const { x = 0, width = 0, payload, ...rest } = props;
  
  // 检查必要的数据
  if (!payload || !payload.open || !payload.close || !payload.high || !payload.low) {
    return null;
  }

  const { open, close, high, low } = payload;
  
  // 从props中获取yAxis的scale函数（Recharts会自动传递）
  // 如果没有，尝试从其他props中获取
  let yScale: any = null;
  
  // 尝试多种方式获取Y轴的scale
  if (rest.yAxis) {
    yScale = rest.yAxis.scale;
  } else if (rest.yAxisMap) {
    const yAxisId = rest.yAxisId || 0;
    yScale = rest.yAxisMap[yAxisId]?.scale;
  }
  
  // 如果没有yScale，尝试从其他props计算
  if (!yScale && rest.viewBox) {
    const { y: viewY, height: viewHeight } = rest.viewBox;
    // 简单的线性映射
    const domain = rest.domain || [0, 100];
    const [minDomain, maxDomain] = domain;
    yScale = (value: number) => {
      const ratio = (maxDomain - value) / (maxDomain - minDomain);
      return viewY + ratio * viewHeight;
    };
  }
  
  // 如果还是没有yScale，返回null
  if (!yScale) {
    console.warn('CandlestickShape: No yScale available');
    return null;
  }
  
  // 判断涨跌
  const isRising = close >= open;
  const color = isRising ? '#ef4444' : '#22c55e'; // 红涨绿跌
  
  // 计算蜡烛图各部分的位置
  const candleWidth = Math.max(width * 0.6, 2); // 蜡烛主体宽度
  const centerX = x + width / 2;
  
  // 使用yScale计算Y坐标
  const highY = yScale(high);
  const lowY = yScale(low);
  const openY = yScale(open);
  const closeY = yScale(close);
  
  const bodyTop = Math.min(openY, closeY);
  const bodyBottom = Math.max(openY, closeY);
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);
  
  return (
    <g>
      {/* 上影线 */}
      <line
        x1={centerX}
        y1={highY}
        x2={centerX}
        y2={bodyTop}
        stroke={color}
        strokeWidth={1}
      />
      
      {/* 蜡烛主体 */}
      <rect
        x={centerX - candleWidth / 2}
        y={bodyTop}
        width={candleWidth}
        height={bodyHeight}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
      
      {/* 下影线 */}
      <line
        x1={centerX}
        y1={bodyBottom}
        x2={centerX}
        y2={lowY}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
};

export default CandlestickShape;
