import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

const amounts = [50, 100, 150, 200];

export function GiftCardsPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.simple.giftCrumb") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <h1 className="text-2xl font-bold text-slate-900">{t("pages.simple.giftTitle")}</h1>
        <p className="text-sm text-slate-600">{t("pages.simple.giftBlurb")}</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {amounts.map((amt) => (
            <button key={amt} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-900 hover:border-primary-300 hover:bg-primary-50">
              ${amt}
            </button>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <input placeholder={t("pages.simple.recipient")} className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
          <input placeholder={t("pages.simple.yourName")} className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
          <input placeholder={t("pages.simple.giftMessage")} className="rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none sm:col-span-2" />
        </div>
        <button className="w-full sm:w-auto rounded-full bg-primary-600 px-6 py-3 text-base font-bold text-white hover:bg-primary-700">{t("pages.simple.sendGift")}</button>
      </div>
    </div>
  );
}
