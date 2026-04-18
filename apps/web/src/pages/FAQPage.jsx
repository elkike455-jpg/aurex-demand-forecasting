import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

export function FAQPage() {
  const { t } = useLanguage();
  const faqs = t("pages.faq.items");
  const [open, setOpen] = useState(faqs[0].q);

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.faq.crumb") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-4">{t("pages.faq.title")}</h1>
        <div className="divide-y divide-slate-100">
          {faqs.map((item) => (
            <button
              key={item.q}
              onClick={() => setOpen(open === item.q ? "" : item.q)}
              className="w-full text-left py-4 focus:outline-none"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-base font-semibold text-slate-900">{item.q}</span>
                <span className="text-primary-500 text-lg">{open === item.q ? "-" : "+"}</span>
              </div>
              {open === item.q && (
                <p className="mt-2 text-sm text-slate-600 leading-relaxed">{item.a}</p>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
