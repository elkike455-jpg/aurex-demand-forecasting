import { Breadcrumbs } from "../components/Breadcrumbs";

export function TermsPage() {
  return (
    <div className="space-y-6">
      <Breadcrumbs items={[{ label: "Terms" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-700 leading-relaxed">
        <h1 className="text-2xl font-bold text-slate-900">Terms of service</h1>
        <p>By shopping with AUREX you agree to our policies on payments, returns, and acceptable use.</p>
        <ul className="list-disc pl-5 space-y-2">
          <li>Orders are processed upon authorization of payment.</li>
          <li>Some items may have manufacturer-specific warranties; details shown on product pages.</li>
          <li>Misuse, fraud, or chargeback abuse may result in account suspension.</li>
        </ul>
        <p>For full terms or legal inquiries, contact legal@aurex.com.</p>
      </div>
    </div>
  );
}
