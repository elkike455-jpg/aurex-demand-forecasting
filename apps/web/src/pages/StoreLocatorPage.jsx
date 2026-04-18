import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

const stores = [
  { city: "New York", address: "135 Mercer St", hours: "10am - 7pm", phone: "+1 (212) 555-0110" },
  { city: "Los Angeles", address: "742 Sunset Blvd", hours: "10am - 8pm", phone: "+1 (310) 555-0199" },
  { city: "Austin", address: "221 South Congress Ave", hours: "10am - 7pm", phone: "+1 (512) 555-0144" },
];

export function StoreLocatorPage() {
  const { t } = useLanguage();

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.simple.storesCrumb") }]} />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {stores.map((store) => (
          <div key={store.city} className="rounded-2xl border-2 border-slate-200 bg-white p-5 shadow-sm space-y-2">
            <h2 className="text-xl font-bold text-slate-900">{store.city}</h2>
            <p className="text-sm text-slate-700">{store.address}</p>
            <p className="text-sm text-slate-600">{t("pages.simple.hours")} {store.hours}</p>
            <p className="text-sm text-slate-600">{t("pages.simple.phone")} {store.phone}</p>
            <button className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">{t("pages.simple.viewDetails")}</button>
          </div>
        ))}
      </div>
    </div>
  );
}
