import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

const roles = [
  { title: "Product Designer", type: "Remote", level: "Mid-Senior" },
  { title: "Frontend Engineer", type: "Remote", level: "Senior" },
  { title: "Customer Experience Lead", type: "US", level: "Mid" },
];

export function CareersPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.simple.careersCrumb") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <h1 className="text-2xl font-bold text-slate-900">{t("pages.simple.careersTitle")}</h1>
        <p className="text-sm text-slate-600">{t("pages.simple.careersBlurb")}</p>
        <div className="space-y-3">
          {roles.map((role) => (
            <div key={role.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">{role.title}</h3>
                <p className="text-sm text-slate-600">{role.type} - {role.level}</p>
              </div>
              <button className="rounded-full bg-primary-600 px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700">{t("pages.simple.apply")}</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
