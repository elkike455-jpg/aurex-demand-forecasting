import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";

export function AboutPage() {
  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "About" }]} />
      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-4">
          <p className="text-primary-600 font-semibold text-sm">Our story</p>
          <h1 className="text-3xl font-bold text-slate-900">Design-forward marketplace for modern living.</h1>
          <p className="text-slate-600 leading-relaxed">
            AUREX curates tech, furniture, and lifestyle products that feel premium without the friction. We prioritize fast shipping,
            transparent policies, and support that responds in minutes.
          </p>
          <div className="grid grid-cols-2 gap-4 text-sm text-slate-700">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">24-48h</div>
              <div>Average ship window</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">4.8/5</div>
              <div>Shopper rating</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">30 days</div>
              <div>Hassle-free returns</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">200+</div>
              <div>Brands curated</div>
            </div>
          </div>
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-xl font-bold text-slate-900">What we stand for</h2>
          <ul className="space-y-3 text-slate-700 text-sm">
            <li className="flex gap-3"><span className="text-primary-600">?</span>Fast, accurate shipping info before checkout.</li>
            <li className="flex gap-3"><span className="text-primary-600">?</span>Transparent policies: returns, warranty, and repairs.</li>
            <li className="flex gap-3"><span className="text-primary-600">?</span>Curated catalog — no endless scroll.</li>
            <li className="flex gap-3"><span className="text-primary-600">?</span>Support that answers within minutes.</li>
          </ul>
          <div className="pt-4">
            <TrustBadges compact />
          </div>
        </div>
      </div>
    </div>
  );
}
