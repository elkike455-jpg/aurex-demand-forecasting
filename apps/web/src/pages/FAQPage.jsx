import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";

const FAQS = [
  { q: "What are your shipping times?", a: "Most items ship in 24–48 hours. Delivery estimates show before checkout." },
  { q: "How do returns work?", a: "30-day window. Start a return from your account or reach us via chat for a prepaid label." },
  { q: "Is checkout secure?", a: "Yes, we use SSL and tokenized payments. We never store raw card data." },
  { q: "Can I track my order?", a: "Use the Track Order page with your email and order ID for real-time updates." },
  { q: "Do you price match?", a: "We review price-match requests within 24h. Chat with support with a link to the lower price." },
];

export function FAQPage() {
  const [open, setOpen] = useState(FAQS[0].q);

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Help / FAQ" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-4">Frequently asked questions</h1>
        <div className="divide-y divide-slate-100">
          {FAQS.map((item) => (
            <button
              key={item.q}
              onClick={() => setOpen(open === item.q ? "" : item.q)}
              className="w-full text-left py-4 focus:outline-none"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-base font-semibold text-slate-900">{item.q}</span>
                <span className="text-primary-500 text-lg">{open === item.q ? "-" : "+"}</span>
              </div>
              {open === item.q && (
                <p className="mt-2 text-sm text-slate-600 leading-relaxed">{item.a}</p>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
