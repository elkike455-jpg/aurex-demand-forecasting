import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";
import { useLanguage } from "../context/LanguageContext";

export function AboutPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.about.crumb") }]} />
      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-4">
          <p className="text-primary-600 font-semibold text-sm">{t("pages.about.eyebrow")}</p>
          <h1 className="text-3xl font-bold text-slate-900">{t("pages.about.title")}</h1>
          <p className="text-slate-600 leading-relaxed">{t("pages.about.body")}</p>
          <div className="grid grid-cols-2 gap-4 text-sm text-slate-700">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">24-48h</div>
              <div>{t("pages.about.statShip")}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">4.8/5</div>
              <div>{t("pages.about.statRating")}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">30 days</div>
              <div>{t("pages.about.statReturns")}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-2xl font-bold text-slate-900">200+</div>
              <div>{t("pages.about.statBrands")}</div>
            </div>
          </div>
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-xl font-bold text-slate-900">{t("pages.about.valuesTitle")}</h2>
          <ul className="space-y-3 text-slate-700 text-sm">
            {t("pages.about.values").map((value) => (
              <li className="flex gap-3" key={value}><span className="text-primary-600">OK</span>{value}</li>
            ))}
          </ul>
          <div className="pt-4">
            <TrustBadges compact />
          </div>
        </div>
      </div>
    </div>
  );
}
