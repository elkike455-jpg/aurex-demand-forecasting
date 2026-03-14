import { Breadcrumbs } from "../components/Breadcrumbs";

export function ShippingReturnsPage() {
  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Shipping & returns" }]} />
      <div className="grid gap-8 md:grid-cols-2">
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <h1 className="text-2xl font-bold text-slate-900">Shipping</h1>
          <p className="text-sm text-slate-600">Real-time delivery estimates are shown at checkout based on your address.</p>
          <ul className="space-y-2 text-sm text-slate-700">
            <li><span className="font-semibold">Processing:</span> 24–48 hours for most items.</li>
            <li><span className="font-semibold">Carriers:</span> UPS, FedEx, DHL. Tracking number emailed instantly.</li>
            <li><span className="font-semibold">Free shipping:</span> Orders $99+ in contiguous US.</li>
            <li><span className="font-semibold">Oversized:</span> White-glove delivery available at checkout.</li>
          </ul>
        </section>
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <h2 className="text-2xl font-bold text-slate-900">Returns</h2>
          <p className="text-sm text-slate-600">30-day hassle-free returns on most items.</p>
          <ul className="space-y-2 text-sm text-slate-700">
            <li>Start a return from your account or via <strong>Track order</strong>.</li>
            <li>Prepaid labels for eligible items; restocking only on opened electronics (10%).</li>
            <li>Refunds issued to original payment method within 3–5 business days after inspection.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
