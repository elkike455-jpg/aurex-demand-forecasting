import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

export function ShippingReturnsPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.simple.shippingCrumb") }]} />
      <div className="grid gap-8 md:grid-cols-2">
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <h1 className="text-2xl font-bold text-slate-900">{t("pages.simple.shippingTitle")}</h1>
          <p className="text-sm text-slate-600">{t("pages.simple.shippingBlurb")}</p>
          <ul className="space-y-2 text-sm text-slate-700">
            {t("pages.simple.shippingBullets").map((item) => <li key={item}>{item}</li>)}
          </ul>
        </section>
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <h2 className="text-2xl font-bold text-slate-900">{t("pages.simple.returnsTitle")}</h2>
          <p className="text-sm text-slate-600">{t("pages.simple.returnsBlurb")}</p>
          <ul className="space-y-2 text-sm text-slate-700">
            {t("pages.simple.returnsBullets").map((item) => <li key={item}>{item}</li>)}
          </ul>
        </section>
      </div>
    </div>
  );
}
