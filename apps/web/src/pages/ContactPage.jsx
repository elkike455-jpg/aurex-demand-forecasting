import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";

export function ContactPage() {
  const [status, setStatus] = useState("idle");

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("sent");
    setTimeout(() => setStatus("idle"), 2500);
  };

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Contact" }]} />
      <div className="grid gap-8 md:grid-cols-[1.1fr,0.9fr]">
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-bold text-slate-900 mb-3">Contact us</h1>
          <p className="text-sm text-slate-600 mb-6">We reply in under 1 hour during business hours.</p>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <input required name="name" placeholder="Full name" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input required type="email" name="email" placeholder="Email" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input name="order" placeholder="Order number (optional)" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <textarea required name="message" rows="4" placeholder="How can we help?" className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <button type="submit" className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition">
              {status === "sent" ? "Message sent" : "Send message"}
            </button>
          </form>
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-xl font-bold text-slate-900">Support channels</h2>
          <ul className="space-y-3 text-sm text-slate-700">
            <li><span className="font-semibold text-slate-900">Live chat:</span> 24/7 (average reply 2 min)</li>
            <li><span className="font-semibold text-slate-900">Email:</span> support@aurex.com</li>
            <li><span className="font-semibold text-slate-900">Phone:</span> +1 (555) 123-9876</li>
            <li><span className="font-semibold text-slate-900">Press:</span> press@aurex.com</li>
          </ul>
          <div className="pt-2">
            <TrustBadges compact />
          </div>
        </div>
      </div>
    </div>
  );
}
