import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

export function PrivacyPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-6">
      <Breadcrumbs items={[{ label: t("pages.simple.privacyCrumb") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4 text-sm text-slate-700 leading-relaxed">
        <h1 className="text-2xl font-bold text-slate-900">{t("pages.simple.privacyTitle")}</h1>
        <p>{t("pages.simple.privacyBody")}</p>
      </div>
    </div>
  );
}
