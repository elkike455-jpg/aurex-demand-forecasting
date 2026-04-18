import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ProductCard } from "../components/ProductCard";
import { TrustBadges } from "../components/TrustBadges";
import { useLanguage } from "../context/LanguageContext";
import { useCommerce } from "../context/CommerceContext";
import { getCategoryVisual } from "../data/categoryVisuals";

const categoryAccents = [
  "from-sky-500/20 to-transparent",
  "from-amber-500/20 to-transparent",
  "from-emerald-500/20 to-transparent",
  "from-primary-500/20 to-transparent",
  "from-cyan-500/20 to-transparent",
  "from-indigo-500/20 to-transparent",
  "from-orange-500/20 to-transparent",
  "from-lime-500/20 to-transparent",
];

const mustHavePages = [
  { labelKey: "footer.about", to: "/about" },
  { labelKey: "footer.contact", to: "/contact" },
  { labelKey: "footer.faq", to: "/help" },
  { labelKey: "mustHavePages.shipping", to: "/shipping-returns" },
  { labelKey: "mustHavePages.returns", to: "/shipping-returns" },
  { labelKey: "footer.privacy", to: "/privacy" },
  { labelKey: "footer.terms", to: "/terms" },
  { labelKey: "footer.giftCards", to: "/gift-cards" },
  { labelKey: "footer.trackOrder", to: "/track-order" },
  { labelKey: "mustHavePages.blog", to: "/blog" },
  { labelKey: "footer.careers", to: "/careers" },
  { labelKey: "footer.storeLocator", to: "/stores" },
];

export function HomePage() {
  const [speed, setSpeed] = useState("normal");
  const { t } = useLanguage();
  const { products, categories: commerceCategories } = useCommerce();

  const speedMap = useMemo(
    () => ({
      slow: "42s",
      normal: "30s",
      fast: "20s",
    }),
    []
  );

  const categories = useMemo(
    () =>
      commerceCategories.map((category, index) => ({
        ...category,
        title: category.name,
        blurb: category.description,
        accent: categoryAccents[index % categoryAccents.length],
      })),
    [commerceCategories]
  );
  const categoryLoop = useMemo(() => [...categories, ...categories], [categories]);

  return (
    <div className="space-y-12">
      <section className="relative overflow-hidden rounded-3xl border-2 border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-primary-800 text-slate-50 shadow-xl">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,rgba(168,85,247,0.35),transparent_45%),radial-gradient(circle_at_80%_20%,rgba(56,189,248,0.25),transparent_45%)]" />
        <div className="relative grid gap-8 lg:grid-cols-[1.3fr,1fr] p-8 md:p-12">
          <div className="space-y-6">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm font-semibold ring-1 ring-white/30">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              {t("home.badge")}
            </span>
            <h1 className="text-3xl md:text-5xl font-black leading-tight">
              {t("home.headline")}
              <span className="block text-primary-100">{t("home.subheadline")}</span>
            </h1>
            <p className="text-base md:text-lg text-slate-200/90 max-w-2xl leading-relaxed">
              {t("home.intro")}
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <Link
                to="/categories"
                className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-7 py-3 text-base font-bold text-white shadow-lg hover:from-primary-400 hover:to-sky-400"
              >
                {t("home.shopCollection")}
              </Link>
              <Link
                to="/products"
                className="rounded-full border-2 border-white/40 px-6 py-3 text-sm font-semibold text-white hover:border-primary-200"
              >
                {t("home.viewDeals")}
              </Link>
              <div className="flex items-center gap-3 text-sm text-slate-200">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-sm font-bold">Top</span>
                <span>
                  {t("home.rating")}
                  <br />
                  {t("home.ships")}
                </span>
              </div>
            </div>
          </div>

          <div className="relative rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur">
            <div className="grid grid-cols-2 gap-3 text-slate-100 text-sm">
              {t("home.pills").map((pill) => (
                <span key={pill} className="rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-center font-semibold">
                  {pill}
                </span>
              ))}
            </div>
            <div className="mt-6 rounded-2xl bg-white p-4 text-slate-900 shadow-lg">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold text-primary-600">{t("home.trendingPick")}</p>
                  <h3 className="text-lg font-bold">{t("home.bundleTitle")}</h3>
                  <p className="text-sm text-slate-600">{t("home.bundlePrice")}</p>
                </div>
                <span className="rounded-full bg-primary-100 px-3 py-1 text-xs font-bold text-primary-700">{t("home.fastShip")}</span>
              </div>
              <div className="mt-3 flex items-center justify-between">
                <div className="text-sm text-slate-600">
                  <div className="font-bold text-slate-900 text-xl">$59.99</div>
                  <div>{t("home.delivery")}</div>
                </div>
                <Link
                  to="/product/4"
                  className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                >
                  {t("home.viewBundle")}
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-900">{t("home.shopByCategory")}</h2>
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center rounded-full border border-slate-200 p-1 bg-slate-50">
              {["slow", "normal", "fast"].map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setSpeed(option)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold capitalize transition-colors ${
                    speed === option ? "bg-primary-600 text-white" : "text-slate-600 hover:text-primary-600"
                  }`}
                >
                  {t(`home.${option}`)}
                </button>
              ))}
            </div>
            <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
              {t("home.viewAll")}
            </Link>
          </div>
        </div>
        <div className="overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_12%,black_88%,transparent)]">
          <div
            className="category-track flex gap-3 w-max hover:[animation-play-state:paused]"
            style={{ "--category-speed": speedMap[speed] }}
          >
            {categoryLoop.map((category, index) => (
              <Link
                key={`${category.title}-${index}`}
                to={`/category/${category.slug}`}
                className="group relative min-w-[220px] overflow-hidden rounded-xl border border-slate-200 bg-slate-900 px-4 py-4 text-left text-white shadow-sm transition hover:-translate-y-0.5 hover:border-primary-300"
              >
                <img src={getCategoryVisual(category).image} alt={category.title} className="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-105" />
                <div className="absolute inset-0 bg-slate-950/55" />
                <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${category.accent} opacity-80`} />
                <div className="relative">
                  <p className="text-sm font-bold text-white">{category.title}</p>
                  <p className="mt-1 text-xs text-slate-100">{category.blurb}</p>
                  <p className="mt-3 text-xs font-semibold text-cyan-200">{t("home.exploreNow")}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
        <h2 className="text-lg font-bold text-slate-900 mb-6 text-center">{t("home.whyShop")}</h2>
        <TrustBadges />
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
        <div className="flex flex-wrap items-start gap-4 justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900">{t("home.mustHave")}</h2>
            <p className="text-sm text-slate-600">{t("home.mustHaveBlurb")}</p>
          </div>
          <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">{t("home.seeAllProducts")}</Link>
        </div>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
          {mustHavePages.map((page) => (
            <Link
              key={page.labelKey}
              to={page.to}
              className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800 hover:border-primary-200 hover:bg-primary-50"
            >
              <span className="h-6 w-6 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs">{t("home.go")}</span>
              <span>{t(page.labelKey)}</span>
              <span className="ml-auto text-xs text-primary-600">{t("home.open")}</span>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-center justify-between gap-4">
          <h2 className="text-xl md:text-2xl font-bold text-slate-900">{t("home.recommended")}</h2>
          <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
            {t("home.viewAllProducts")}
          </Link>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
          {products.filter((product) => product.active).slice(0, 4).map((product) => (
            <ProductCard product={product} key={product.id} />
          ))}
        </div>
      </section>
    </div>
  );
}
