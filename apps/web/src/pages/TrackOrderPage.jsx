import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";

export function TrackOrderPage() {
  const [status, setStatus] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus({
      eta: "Arrives in 2–4 days",
      carrier: "UPS",
      lastUpdate: "Label created, awaiting pickup",
    });
  };

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Track order" }]} />
      <div className="grid gap-8 lg:grid-cols-[1.2fr,1fr]">
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-bold text-slate-900 mb-3">Track your order</h1>
          <p className="text-sm text-slate-600 mb-4">Enter your order ID and email to see live status.</p>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <input required name="orderId" placeholder="Order ID" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input required type="email" name="email" placeholder="Email" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <button type="submit" className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition">
              Check status
            </button>
          </form>
          {status && (
            <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              <div className="font-semibold text-slate-900">{status.eta}</div>
              <p>{status.carrier}</p>
              <p className="text-slate-500">{status.lastUpdate}</p>
            </div>
          )}
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-700">
          <h2 className="text-xl font-bold text-slate-900">Return or exchange</h2>
          <p>Use your order ID to start a return. Most items are eligible within 30 days.</p>
          <ul className="list-disc pl-5 space-y-1">
            <li>Generate a prepaid label for eligible items.</li>
            <li>Schedule pickup or drop-off at partner locations.</li>
            <li>Refund issued within 3–5 business days after inspection.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
