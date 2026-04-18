import { ForecastHookCard, StatCard } from "../components/AdminCards";
import { useCommerce } from "../../context/CommerceContext";

export function AdminAnalyticsPage() {
  const { analytics } = useCommerce();

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-black text-slate-950">Analytics</h1>
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard label="Forecast coverage" value="Ready" detail="Components and admin routes prepared" />
        <StatCard label="Low stock signals" value={analytics.lowStockProducts} detail="Inventory features available now" tone="amber" />
        <StatCard label="Model API" value="Pending" detail="Connect services/api forecast endpoint later" tone="navy" />
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <ForecastHookCard title="Demand insights card" />
        <ForecastHookCard title="Stockout probability chart" />
        <ForecastHookCard title="SME reorder assistant" />
      </div>
    </div>
  );
}
