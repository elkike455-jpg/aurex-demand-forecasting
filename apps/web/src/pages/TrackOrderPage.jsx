import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

export function TrackOrderPage() {
  const [status, setStatus] = useState(null);
  const { t } = useLanguage();

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus(true);
  };

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.simple.trackCrumb") }]} />
      <div className="grid gap-8 lg:grid-cols-[1.2fr,1fr]">
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-bold text-slate-900 mb-3">{t("pages.simple.trackTitle")}</h1>
          <p className="text-sm text-slate-600 mb-4">{t("pages.simple.trackBlurb")}</p>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <input required name="orderId" placeholder={t("pages.simple.trackCrumb")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input required type="email" name="email" placeholder={t("checkout.email")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <button type="submit" className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition">
              {t("pages.simple.checkStatus")}
            </button>
          </form>
          {status && (
            <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              <div className="font-semibold text-slate-900">2-4 days</div>
              <p>UPS</p>
            </div>
          )}
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-700">
          <h2 className="text-xl font-bold text-slate-900">{t("pages.simple.returnExchange")}</h2>
          <p>{t("pages.simple.returnBody")}</p>
        </div>
      </div>
    </div>
  );
}
