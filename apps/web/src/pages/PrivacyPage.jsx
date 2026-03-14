import { Breadcrumbs } from "../components/Breadcrumbs";

export function PrivacyPage() {
  return (
    <div className="space-y-6">
      <Breadcrumbs items={[{ label: "Privacy" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-700 leading-relaxed">
        <h1 className="text-2xl font-bold text-slate-900">Privacy policy</h1>
        <p>We collect only the data needed to process orders, provide support, and improve your experience. We never sell personal data.</p>
        <ul className="list-disc pl-5 space-y-2">
          <li>Payments are tokenized; we do not store raw card data.</li>
          <li>Analytics are anonymized and opt-out is available in account settings.</li>
          <li>You can request data export or deletion at any time via support.</li>
        </ul>
        <p>Contact privacy@aurex.com for any questions.</p>
      </div>
    </div>
  );
}
