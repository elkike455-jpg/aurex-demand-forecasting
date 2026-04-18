import { useState } from "react";
import { Header } from "../components/Header";
import { TrustBadges } from "../components/TrustBadges";
import { MiniCartDrawer } from "../components/MiniCartDrawer";
import { Link } from "react-router-dom";
import { useLanguage } from "../context/LanguageContext";

export function AppLayout({ children }) {
  const [isCartOpen, setIsCartOpen] = useState(false);
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <Header onCartOpen={() => setIsCartOpen(true)} />
      <main className="max-w-6xl mx-auto w-full px-5 py-8 flex-1">{children}</main>
      <MiniCartDrawer open={isCartOpen} onClose={() => setIsCartOpen(false)} />
      <footer className="border-t-2 border-slate-200 bg-white mt-auto">
        <div className="max-w-6xl mx-auto px-5 py-10">
          <div className="grid gap-8 md:grid-cols-4 mb-10">
            <div>
              <img src="/aurexlogo-transparent.png" alt="AUREX" className="mb-3 h-20 w-24 object-contain" />
              <p className="text-sm text-slate-600">
                {t("footer.blurb")}
              </p>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">{t("footer.shop")}</div>
              <ul className="space-y-2 text-sm">
                <li><Link to="/products" className="text-slate-600 hover:text-primary-600">{t("footer.allProducts")}</Link></li>
                <li><Link to="/products" className="text-slate-600 hover:text-primary-600">{t("footer.deals")}</Link></li>
                <li><Link to="/gift-cards" className="text-slate-600 hover:text-primary-600">{t("footer.giftCards")}</Link></li>
                <li><Link to="/stores" className="text-slate-600 hover:text-primary-600">{t("footer.storeLocator")}</Link></li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">{t("footer.support")}</div>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><Link to="/contact" className="hover:text-primary-600">{t("footer.contact")}</Link></li>
                <li><Link to="/help" className="hover:text-primary-600">{t("footer.faq")}</Link></li>
                <li><Link to="/shipping-returns" className="hover:text-primary-600">{t("footer.shippingReturns")}</Link></li>
                <li><Link to="/track-order" className="hover:text-primary-600">{t("footer.trackOrder")}</Link></li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">{t("footer.legal")}</div>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><Link to="/privacy" className="hover:text-primary-600">{t("footer.privacy")}</Link></li>
                <li><Link to="/terms" className="hover:text-primary-600">{t("footer.terms")}</Link></li>
                <li><Link to="/about" className="hover:text-primary-600">{t("footer.about")}</Link></li>
                <li><Link to="/careers" className="hover:text-primary-600">{t("footer.careers")}</Link></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-top border-slate-200 flex flex-wrap items-center justify-between gap-4 text-sm text-slate-600">
            <span>© {new Date().getFullYear()} AUREX. {t("footer.rights")}</span>
            <TrustBadges compact />
          </div>
        </div>
      </footer>
    </div>
  );
}

