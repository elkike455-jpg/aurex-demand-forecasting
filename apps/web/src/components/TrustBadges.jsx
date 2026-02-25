export function TrustBadges({ compact = false }) {
  const badges = [
    { icon: "🔒", label: "Secure checkout", sub: "SSL encrypted" },
    { icon: "🚚", label: "Free shipping", sub: "On orders $99+" },
    { icon: "↩️", label: "Easy returns", sub: "30-day guarantee" },
  ];

  if (compact) {
    return (
      <div className="flex flex-wrap items-center justify-center gap-6 text-slate-600 text-sm">
        {badges.map((b) => (
          <span key={b.label} className="flex items-center gap-2 font-medium">
            <span className="text-lg" aria-hidden>{b.icon}</span>
            <span>{b.label}</span>
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
      {badges.map((b) => (
        <div
          key={b.label}
          className="rounded-xl border-2 border-slate-100 bg-white px-4 py-4 shadow-sm transition hover:border-primary-100 hover:shadow-md"
        >
          <span className="text-2xl block mb-2" aria-hidden>{b.icon}</span>
          <div className="font-bold text-slate-900">{b.label}</div>
          <div className="text-sm text-slate-500">{b.sub}</div>
        </div>
      ))}
    </div>
  );
}
