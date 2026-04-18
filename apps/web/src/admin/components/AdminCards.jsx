export function StatCard({ label, value, color = "#1976d2", icon = "▣" }) {
  return (
    <div className="aurex-widget" style={{ backgroundColor: color }}>
      <div>
        <div className="aurex-widget-label">{label}</div>
        <div className="aurex-widget-value">{value}</div>
      </div>
      <div className="aurex-widget-icon" aria-hidden>{icon}</div>
    </div>
  );
}

export function StatusBadge({ status }) {
  return <span className="aurex-status">{status}</span>;
}

export function EmptyState({ title, body }) {
  return (
    <div className="aurex-form-card" style={{ textAlign: "center" }}>
      <h3 style={{ fontWeight: 900 }}>{title}</h3>
      <p style={{ color: "#64748b", marginTop: 8 }}>{body}</p>
    </div>
  );
}

export function ForecastHookCard({ title = "Forecast intelligence hook" }) {
  return (
    <div className="aurex-form-card">
      <h3 style={{ color: "#1976d2", fontWeight: 900, textAlign: "center" }}>{title}</h3>
      <p style={{ color: "#64748b", marginTop: 10, lineHeight: 1.7 }}>
        Future AUREX forecasting slot for demand curves, reorder recommendations, stockout risk, and model confidence.
      </p>
    </div>
  );
}
