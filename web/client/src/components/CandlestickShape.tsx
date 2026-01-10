import React from 'react';

interface CandlestickShapeProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: any;
}

/**
 * 自定义蜡烛图形状组件
 * 红色：收盘价 > 开盘价（上涨）
 * 绿色：收盘价 < 开盘价（下跌）
 */
export const CandlestickShape: React.FC<CandlestickShapeProps> = (props) => {
  const { x = 0, y = 0, width = 0, height = 0, payload } = props;
  
  if (!payload || !payload.open || !payload.close || !payload.high || !payload.low) {
    return null;
  }

  const { open, close, high, low } = payload;
  
  // 计算Y轴比例（基于图表高度和价格范围）
  const priceRange = Math.max(high, open, close) - Math.min(low, open, close);
  if (priceRange === 0) return null;
  
  // 判断涨跌
  const isRising = close >= open;
  const color = isRising ? '#ef4444' : '#22c55e'; // 红涨绿跌
  
  // 计算蜡烛图各部分的位置
  const candleWidth = Math.max(width * 0.6, 2); // 蜡烛主体宽度
  const centerX = x + width / 2;
  
  // 计算价格对应的Y坐标（需要根据图表的Y轴域来计算）
  // 这里简化处理，使用相对位置
  const maxPrice = Math.max(high, open, close);
  const minPrice = Math.min(low, open, close);
  const bodyTop = Math.max(open, close);
  const bodyBottom = Math.min(open, close);
  
  // 计算相对高度
  const totalHeight = height;
  const bodyHeight = Math.max(((bodyTop - bodyBottom) / priceRange) * totalHeight, 1);
  const topWickHeight = ((maxPrice - bodyTop) / priceRange) * totalHeight;
  const bottomWickHeight = ((bodyBottom - minPrice) / priceRange) * totalHeight;
  
  // 计算Y坐标（从上到下）
  const topWickY = y;
  const bodyY = topWickY + topWickHeight;
  const bottomWickY = bodyY + bodyHeight;
  
  return (
    <g>
      {/* 上影线 */}
      <line
        x1={centerX}
        y1={topWickY}
        x2={centerX}
        y2={bodyY}
        stroke={color}
        strokeWidth={1}
      />
      
      {/* 蜡烛主体 */}
      <rect
        x={centerX - candleWidth / 2}
        y={bodyY}
        width={candleWidth}
        height={bodyHeight}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
      
      {/* 下影线 */}
      <line
        x1={centerX}
        y1={bottomWickY}
        x2={centerX}
        y2={bottomWickY + bottomWickHeight}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
};

export default CandlestickShape;
