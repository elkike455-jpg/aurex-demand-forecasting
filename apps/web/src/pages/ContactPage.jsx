import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";
import { useLanguage } from "../context/LanguageContext";

export function ContactPage() {
  const [status, setStatus] = useState("idle");
  const { t } = useLanguage();

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("sent");
    setTimeout(() => setStatus("idle"), 2500);
  };

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("pages.contact.crumb") }]} />
      <div className="grid gap-8 md:grid-cols-[1.1fr,0.9fr]">
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-bold text-slate-900 mb-3">{t("pages.contact.title")}</h1>
          <p className="text-sm text-slate-600 mb-6">{t("pages.contact.blurb")}</p>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <input required name="name" placeholder={t("pages.contact.name")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input required type="email" name="email" placeholder={t("pages.contact.email")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <input name="order" placeholder={t("pages.contact.order")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <textarea required name="message" rows="4" placeholder={t("pages.contact.message")} className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none" />
            <button type="submit" className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition">
              {status === "sent" ? t("pages.contact.sent") : t("pages.contact.send")}
            </button>
          </form>
        </div>
        <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-xl font-bold text-slate-900">{t("pages.contact.channels")}</h2>
          <ul className="space-y-3 text-sm text-slate-700">
            <li><span className="font-semibold text-slate-900">{t("pages.contact.liveChat")}</span> {t("pages.contact.liveChatValue")}</li>
            <li><span className="font-semibold text-slate-900">{t("pages.contact.emailLabel")}</span> support@aurex.com</li>
            <li><span className="font-semibold text-slate-900">{t("pages.contact.phone")}</span> +1 (555) 123-9876</li>
            <li><span className="font-semibold text-slate-900">{t("pages.contact.press")}</span> press@aurex.com</li>
          </ul>
          <div className="pt-2">
            <TrustBadges compact />
          </div>
        </div>
      </div>
    </div>
  );
}
