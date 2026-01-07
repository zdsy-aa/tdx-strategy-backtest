import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";

// Pages
import Home from "./pages/Home";
import Indicators from "./pages/Indicators";
import Strategies from "./pages/Strategies";
import Backtest from "./pages/Backtest";
import AIOptimizer from "./pages/AIOptimizer";
import CustomStrategy from "./pages/CustomStrategy";
import Reports from "./pages/Reports";
import NotFound from "./pages/NotFound";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/indicators" component={Indicators} />
      <Route path="/strategies" component={Strategies} />
      <Route path="/backtest" component={Backtest} />
      <Route path="/ai-optimizer" component={AIOptimizer} />
      <Route path="/custom-strategy" component={CustomStrategy} />
      <Route path="/reports" component={Reports} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
