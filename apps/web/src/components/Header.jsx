import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";

const QUICK_LINKS = [
  { labelKey: "nav.home", to: "/" },
  { labelKey: "nav.shopAll", to: "/products" },
  { labelKey: "nav.categories", to: "/categories" },
  { labelKey: "nav.deals", to: "/products" },
  { labelKey: "nav.giftCards", to: "/gift-cards" },
  { labelKey: "nav.shippingReturns", to: "/shipping-returns" },
  { labelKey: "nav.helpFaq", to: "/help" },
  { labelKey: "nav.trackOrder", to: "/track-order" },
  { labelKey: "nav.stores", to: "/stores" },
];

export function Header({ onCartOpen = () => {} }) {
  const { totalItems } = useCart();
  const { user, isAuthenticated, isStaff, isSeller, signOut } = useAuth();
  const { language, languages, setLanguage, t } = useLanguage();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);

  const suggestions = useMemo(() => {
    const base = t("searchSuggestions");
    if (!query) return base.slice(0, 5);
    return base
      .filter((item) => item.toLowerCase().includes(query.toLowerCase()))
      .slice(0, 5);
  }, [query, t]);

  return (
    <header className="sticky top-0 z-20 border-b bg-white/90 backdrop-blur">
      <div className="max-w-6xl mx-auto px-5 py-4 flex flex-wrap lg:flex-nowrap items-center gap-4 lg:gap-5">
        <Link to="/" className="flex shrink-0 items-center gap-3 cursor-pointer">
          <img
            src="/aurexlogo-transparent.png"
            alt="AUREX"
            className="h-16 w-20 object-contain"
          />
        </Link>

        <div className="min-w-[280px] flex-1 max-w-xl relative">
          <div className="flex items-center gap-2 rounded-full border-2 border-slate-200 bg-slate-50 px-5 py-3 shadow-sm focus-within:border-primary-400 focus-within:bg-white transition-colors">
            <input
              type="text"
              placeholder={t("header.searchPlaceholder")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setTimeout(() => setFocused(false), 120)}
              className="w-full bg-transparent outline-none text-base placeholder:text-slate-400"
            />
            <button className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-5 py-2 text-base font-semibold text-white shadow-sm hover:from-primary-400 hover:to-sky-400 transition-colors">
              {t("header.search")}
            </button>
          </div>
          {focused && suggestions.length > 0 && (
            <div className="absolute left-0 right-0 mt-2 rounded-2xl border border-slate-100 bg-white shadow-xl p-3 text-sm text-slate-700">
              <div className="font-semibold text-slate-900 px-2 pb-2">{t("header.popularSearches")}</div>
              <ul className="divide-y divide-slate-100">
                {suggestions.map((s) => (
                  <li key={s} className="px-2 py-2 hover:bg-slate-50 rounded-xl cursor-pointer">
                    {s}
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-2 px-2 pt-2 text-xs text-slate-500">
                <span aria-hidden>{t("header.top")}</span>
                <span>{t("header.trySearches")}</span>
              </div>
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-3 text-base">
          <label className="flex items-center gap-2 text-sm font-semibold text-slate-600">
            <span className="hidden lg:inline">{t("languageToggleLabel")}</span>
            <select
              aria-label={t("chooseLanguage")}
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              className="rounded-full border-2 border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 focus:border-primary-500 focus:outline-none"
            >
              {languages.map((code) => (
                <option key={code} value={code}>
                  {code.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          {isAuthenticated ? (
            <>
              <div className="hidden md:block text-right leading-tight">
                <p className="text-xs uppercase tracking-[0.12em] text-slate-500 font-semibold">{t("header.account")}</p>
                <Link to="/account" className="font-semibold text-slate-800 hover:text-primary-600">
                  {user.fullName}
                </Link>
              </div>
              <Link to="/account" className="rounded-full border-2 border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:border-primary-400 hover:text-primary-600">
                Mi cuenta
              </Link>
              {isStaff && (
                <Link to="/admin" className="rounded-full border-2 border-cyan-300 px-4 py-2 text-sm font-semibold text-cyan-700 hover:bg-cyan-50">
                  Admin
                </Link>
              )}
              {isSeller && (
                <Link to="/seller/dashboard" className="rounded-full border-2 border-primary-300 px-4 py-2 text-sm font-semibold text-primary-700 hover:bg-primary-50">
                  Vender
                </Link>
              )}
              <button
                onClick={signOut}
                type="button"
                className="rounded-full border-2 border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                {t("header.signOut")}
              </button>
            </>
          ) : (
            <>
              <Link to="/sign-in" className="font-medium text-slate-700 hover:text-primary-600">
                {t("header.signIn")}
              </Link>
              <Link
                to="/create-account"
                className="hidden sm:inline-flex rounded-full border-2 border-slate-300 px-4 py-2 text-sm font-semibold hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                {t("header.createAccount")}
              </Link>
              <Link to="/seller/onboarding" className="hidden sm:inline-flex rounded-full bg-primary-600 px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700">
                Vender
              </Link>
            </>
          )}
          <button
            onClick={onCartOpen}
            className="relative p-2 hover:text-primary-600 text-sm font-semibold"
            aria-label={t("header.cart")}
            type="button"
          >
            {t("header.cart")}
            <span className="absolute -right-2 -top-1 h-6 w-6 rounded-full bg-primary-500 text-sm font-bold text-white flex items-center justify-center shadow-sm">
              {totalItems}
            </span>
          </button>
        </div>
      </div>

      <nav className="border-t bg-white">
        <div className="max-w-6xl mx-auto px-5 py-3 flex gap-4 sm:gap-6 text-sm sm:text-base text-slate-600 overflow-x-auto">
          {QUICK_LINKS.map((item) => (
            <Link
              key={item.labelKey}
              to={item.to}
              className={`shrink-0 whitespace-nowrap font-medium hover:text-primary-600 ${
                item.labelKey === "nav.home" ? "text-slate-900 font-semibold" : ""
              }`}
            >
              {t(item.labelKey)}
            </Link>
          ))}
          <span className="ml-auto hidden lg:inline-flex shrink-0 items-center gap-2 whitespace-nowrap text-xs uppercase tracking-[0.12em] font-semibold text-slate-500">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            {t("nav.limitedBundles")}
          </span>
        </div>
      </nav>
    </header>
  );
}

