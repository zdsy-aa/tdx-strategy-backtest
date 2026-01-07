import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { invokeLLM } from "./_core/llm";
import { z } from "zod";
import fs from "fs";
import path from "path";

// 报告目录路径
const REPORT_DIR = path.join(process.cwd(), "report");
const DATA_DIR = path.join(process.cwd(), "data", "day");

// 策略数据（从 JSON 加载）
const strategiesPath = path.join(process.cwd(), "client", "src", "data", "strategies.json");

export const appRouter = router({
  system: systemRouter,
  
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),

  // ============================================================================
  // 报告相关 API
  // ============================================================================
  reports: router({
    /**
     * 获取报告列表
     * 按类型分类：total（总报告）、yearly（年度）、monthly（月度）
     */
    list: publicProcedure
      .input(z.object({
        type: z.enum(["total", "yearly", "monthly"]).optional(),
      }).optional())
      .query(async ({ input }) => {
        const reports: Array<{
          id: string;
          name: string;
          type: "total" | "yearly" | "monthly";
          path: string;
          updatedAt: string;
        }> = [];

        const types = input?.type ? [input.type] : ["total", "yearly", "monthly"];

        for (const type of types) {
          const dirPath = path.join(REPORT_DIR, type);
          
          if (!fs.existsSync(dirPath)) {
            continue;
          }

          const files = fs.readdirSync(dirPath);
          
          for (const file of files) {
            if (file.endsWith(".md") || file.endsWith(".csv")) {
              const filePath = path.join(dirPath, file);
              const stats = fs.statSync(filePath);
              
              reports.push({
                id: `${type}/${file}`,
                name: file.replace(/\.(md|csv)$/, ""),
                type: type as "total" | "yearly" | "monthly",
                path: `${type}/${file}`,
                updatedAt: stats.mtime.toISOString(),
              });
            }
          }
        }

        return reports.sort((a, b) => 
          new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
        );
      }),

    /**
     * 获取单个报告内容
     */
    get: publicProcedure
      .input(z.object({
        path: z.string(),
      }))
      .query(async ({ input }) => {
        const filePath = path.join(REPORT_DIR, input.path);
        
        if (!fs.existsSync(filePath)) {
          throw new Error(`报告不存在: ${input.path}`);
        }

        const content = fs.readFileSync(filePath, "utf-8");
        const stats = fs.statSync(filePath);
        const isMarkdown = input.path.endsWith(".md");

        return {
          path: input.path,
          content,
          isMarkdown,
          updatedAt: stats.mtime.toISOString(),
        };
      }),
  }),

  // ============================================================================
  // AI 策略优化 API
  // ============================================================================
  ai: router({
    /**
     * AI 策略优化
     * 用户通过自然语言描述需求，AI 给出策略建议
     */
    optimize: publicProcedure
      .input(z.object({
        query: z.string().min(1).max(2000),
        context: z.object({
          riskLevel: z.enum(["conservative", "moderate", "aggressive"]).optional(),
          holdPeriod: z.enum(["short", "medium", "long"]).optional(),
          marketCondition: z.enum(["bull", "bear", "range"]).optional(),
        }).optional(),
      }))
      .mutation(async ({ input }) => {
        // 加载策略数据作为上下文
        let strategiesData = "";
        if (fs.existsSync(strategiesPath)) {
          strategiesData = fs.readFileSync(strategiesPath, "utf-8");
        }

        const systemPrompt = `你是一个专业的量化交易策略顾问。你的任务是根据用户的需求，基于以下指标系统给出交易策略建议。

可用的指标系统：
1. 六脉神剑：MACD、KDJ、RSI、LWR、BBI、MTM 六大技术指标的综合判断
2. 买卖点指标：基于庄家线和散户线的交叉信号
3. 缠论买点：一买、二买、三买、类二买
4. 摇钱树：综合选股信号

策略数据参考：
${strategiesData.substring(0, 5000)}

用户上下文：
- 风险偏好：${input.context?.riskLevel || "未指定"}
- 持有周期：${input.context?.holdPeriod || "未指定"}
- 市场环境：${input.context?.marketCondition || "未指定"}

请根据用户的问题，给出具体的策略建议，包括：
1. 推荐的指标组合
2. 买入条件
3. 卖出条件
4. 仓位建议
5. 风险提示

回复格式要求：使用 Markdown 格式，结构清晰。`;

        const result = await invokeLLM({
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: input.query },
          ],
          maxTokens: 2000,
        });

        const content = result.choices[0]?.message?.content;
        
        return {
          suggestion: typeof content === "string" ? content : JSON.stringify(content),
          model: result.model,
          usage: result.usage,
        };
      }),

    /**
     * AI 策略解读
     * 解读指定策略的详细含义和使用方法
     */
    explain: publicProcedure
      .input(z.object({
        strategyId: z.string(),
      }))
      .mutation(async ({ input }) => {
        let strategiesData: any = { indicators: [], combos: [] };
        if (fs.existsSync(strategiesPath)) {
          strategiesData = JSON.parse(fs.readFileSync(strategiesPath, "utf-8"));
        }

        // 查找策略
        const indicator = strategiesData.indicators?.find((i: any) => i.id === input.strategyId);
        const combo = strategiesData.combos?.find((c: any) => c.id === input.strategyId);
        const strategy = indicator || combo;

        if (!strategy) {
          throw new Error(`策略不存在: ${input.strategyId}`);
        }

        const systemPrompt = `你是一个专业的量化交易策略顾问。请详细解读以下交易策略，包括：
1. 策略原理
2. 适用场景
3. 使用技巧
4. 常见误区
5. 实战案例分析

策略信息：
${JSON.stringify(strategy, null, 2)}

请用通俗易懂的语言解释，帮助用户理解和使用这个策略。`;

        const result = await invokeLLM({
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `请详细解读 "${strategy.name}" 策略` },
          ],
          maxTokens: 2000,
        });

        const content = result.choices[0]?.message?.content;

        return {
          strategyId: input.strategyId,
          strategyName: strategy.name,
          explanation: typeof content === "string" ? content : JSON.stringify(content),
        };
      }),
  }),

  // ============================================================================
  // 自定义策略回测 API
  // ============================================================================
  backtest: router({
    /**
     * 执行自定义策略回测
     */
    run: publicProcedure
      .input(z.object({
        name: z.string().min(1).max(100),
        conditions: z.array(z.object({
          indicator: z.string(),
          operator: z.enum(["eq", "gt", "lt", "gte", "lte"]),
          value: z.union([z.number(), z.boolean()]),
        })),
        holdPeriod: z.number().min(1).max(60),
        offsetDays: z.number().min(0).max(10).default(5),
      }))
      .mutation(async ({ input }) => {
        // 模拟回测结果（实际应调用 Python 脚本）
        // 这里返回模拟数据，实际部署时需要对接 Python 回测引擎
        
        const mockResult = {
          strategyName: input.name,
          conditions: input.conditions,
          holdPeriod: input.holdPeriod,
          offsetDays: input.offsetDays,
          results: {
            signalCount: Math.floor(Math.random() * 100) + 20,
            tradeCount: Math.floor(Math.random() * 80) + 15,
            winRate: (Math.random() * 30 + 45).toFixed(2),
            avgReturn: (Math.random() * 3 - 0.5).toFixed(2),
            maxReturn: (Math.random() * 15 + 5).toFixed(2),
            minReturn: (Math.random() * -10 - 2).toFixed(2),
            totalReturn: (Math.random() * 50 + 10).toFixed(2),
            sharpeRatio: (Math.random() * 1.5 + 0.3).toFixed(2),
          },
          dataRange: {
            start: "2016-01-04",
            end: "2026-01-05",
            tradingDays: 2431,
          },
          generatedAt: new Date().toISOString(),
        };

        return mockResult;
      }),

    /**
     * 获取可用的指标列表
     */
    indicators: publicProcedure.query(() => {
      return [
        { id: "six_veins_count", name: "六脉神剑红色数量", type: "number", range: [0, 6] },
        { id: "macd_red", name: "MACD红", type: "boolean" },
        { id: "kdj_red", name: "KDJ红", type: "boolean" },
        { id: "rsi_red", name: "RSI红", type: "boolean" },
        { id: "lwr_red", name: "LWR红", type: "boolean" },
        { id: "bbi_red", name: "BBI红", type: "boolean" },
        { id: "mtm_red", name: "MTM红", type: "boolean" },
        { id: "buy1", name: "买点1", type: "boolean" },
        { id: "buy2", name: "买点2", type: "boolean" },
        { id: "sell1", name: "卖点1", type: "boolean" },
        { id: "sell2", name: "卖点2", type: "boolean" },
        { id: "chan_buy1", name: "缠论一买", type: "boolean" },
        { id: "chan_buy2", name: "缠论二买", type: "boolean" },
        { id: "chan_buy3", name: "缠论三买", type: "boolean" },
        { id: "money_tree", name: "摇钱树", type: "boolean" },
      ];
    }),
  }),

  // ============================================================================
  // 数据 API
  // ============================================================================
  data: router({
    /**
     * 获取可用的数据文件列表
     */
    list: publicProcedure.query(() => {
      if (!fs.existsSync(DATA_DIR)) {
        return [];
      }

      const files = fs.readdirSync(DATA_DIR);
      return files
        .filter(f => f.endsWith(".csv"))
        .map(f => {
          const filePath = path.join(DATA_DIR, f);
          const stats = fs.statSync(filePath);
          return {
            name: f.replace(".csv", ""),
            file: f,
            size: stats.size,
            updatedAt: stats.mtime.toISOString(),
          };
        });
    }),

    /**
     * 获取数据文件的最新 N 条记录
     */
    preview: publicProcedure
      .input(z.object({
        file: z.string(),
        limit: z.number().min(1).max(100).default(20),
      }))
      .query(({ input }) => {
        const filePath = path.join(DATA_DIR, input.file);
        
        if (!fs.existsSync(filePath)) {
          throw new Error(`数据文件不存在: ${input.file}`);
        }

        const content = fs.readFileSync(filePath, "utf-8");
        const lines = content.trim().split("\n");
        
        if (lines.length < 2) {
          return { headers: [], rows: [] };
        }

        const headers = lines[0].split(",");
        const rows = lines
          .slice(-input.limit - 1, -1)
          .reverse()
          .map(line => {
            const values = line.split(",");
            const row: Record<string, string> = {};
            headers.forEach((h, i) => {
              row[h] = values[i] || "";
            });
            return row;
          });

        return { headers, rows };
      }),
  }),
});

export type AppRouter = typeof appRouter;
