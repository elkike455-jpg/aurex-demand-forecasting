import { Breadcrumbs } from "../components/Breadcrumbs";

const amounts = [50, 100, 150, 200];

export function GiftCardsPage() {
  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Gift cards" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <h1 className="text-2xl font-bold text-slate-900">Send an AUREX gift card</h1>
        <p className="text-sm text-slate-600">Digital delivery within minutes. Choose an amount and personalize your note.</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {amounts.map((amt) => (
            <button key={amt} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-900 hover:border-primary-300 hover:bg-primary-50">
              ${amt}
            </button>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <input placeholder="Recipient email" className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
          <input placeholder="Your name" className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
          <input placeholder="Message (optional)" className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none sm:col-span-2" />
        </div>
        <button className="w-full sm:w-auto rounded-full bg-primary-600 px-6 py-3 text-base font-bold text-white hover:bg-primary-700">Send gift card</button>
      </div>
    </div>
  );
}
