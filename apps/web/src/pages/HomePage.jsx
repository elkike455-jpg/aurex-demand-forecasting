import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { mockProducts } from "../../mocks/products";
import { ProductCard } from "../components/ProductCard";
import { TrustBadges } from "../components/TrustBadges";

const categories = [
  { title: "Furniture", blurb: "Sofas, tables, and storage", accent: "from-sky-500/20 to-transparent" },
  { title: "Lighting & decor", blurb: "Mood lighting and accents", accent: "from-amber-500/20 to-transparent" },
  { title: "Smart home", blurb: "Automation and connected tech", accent: "from-emerald-500/20 to-transparent" },
  { title: "Workspace", blurb: "Desks, chairs, and focus gear", accent: "from-primary-500/20 to-transparent" },
  { title: "Audio & speakers", blurb: "Cinema sound and smart audio", accent: "from-cyan-500/20 to-transparent" },
  { title: "Gaming setup", blurb: "Performance-ready stations", accent: "from-indigo-500/20 to-transparent" },
  { title: "Kitchen & bar", blurb: "Cooking and hosting essentials", accent: "from-orange-500/20 to-transparent" },
  { title: "Outdoor living", blurb: "Patio and balcony upgrades", accent: "from-lime-500/20 to-transparent" },
];

const mustHavePages = [
  { label: "About", to: "/about" },
  { label: "Contact", to: "/contact" },
  { label: "FAQ", to: "/help" },
  { label: "Shipping", to: "/shipping-returns" },
  { label: "Returns", to: "/shipping-returns" },
  { label: "Privacy", to: "/privacy" },
  { label: "Terms", to: "/terms" },
  { label: "Gift cards", to: "/gift-cards" },
  { label: "Track order", to: "/track-order" },
  { label: "Blog", to: "/blog" },
  { label: "Careers", to: "/careers" },
  { label: "Store locator", to: "/stores" },
];

export function HomePage() {
  const [speed, setSpeed] = useState("normal");

  const speedMap = useMemo(
    () => ({
      slow: "42s",
      normal: "30s",
      fast: "20s",
    }),
    []
  );

  const categoryLoop = useMemo(() => [...categories, ...categories], []);

  return (
    <div className="space-y-12">
      <section className="relative overflow-hidden rounded-3xl border-2 border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-primary-800 text-slate-50 shadow-xl">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,rgba(168,85,247,0.35),transparent_45%),radial-gradient(circle_at_80%_20%,rgba(56,189,248,0.25),transparent_45%)]" />
        <div className="relative grid gap-8 lg:grid-cols-[1.3fr,1fr] p-8 md:p-12">
          <div className="space-y-6">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm font-semibold ring-1 ring-white/30">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              New season drop - limited bundles
            </span>
            <h1 className="text-3xl md:text-5xl font-black leading-tight">
              Build the space you want to live in.
              <span className="block text-primary-100">Furniture, lighting, and tech that ships fast.</span>
            </h1>
            <p className="text-base md:text-lg text-slate-200/90 max-w-2xl leading-relaxed">
              Curated gear for modern living. Shop ready-to-ship picks, see delivery times up front, and get effortless returns.
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <Link
                to="/products"
                className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-7 py-3 text-base font-bold text-white shadow-lg hover:from-primary-400 hover:to-sky-400"
              >
                Shop the collection
              </Link>
              <Link
                to="/products"
                className="rounded-full border-2 border-white/40 px-6 py-3 text-sm font-semibold text-white hover:border-primary-200"
              >
                View deals
              </Link>
              <div className="flex items-center gap-3 text-sm text-slate-200">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-lg">Top</span>
                <span>
                  4.8/5 from 2,300+ shoppers
                  <br />
                  Ships in 24-48h
                </span>
              </div>
            </div>
          </div>

          <div className="relative rounded-2xl bg-white/5 border border-white/10 p-6 backdrop-blur">
            <div className="grid grid-cols-2 gap-3 text-slate-100 text-sm">
              {["Home essentials", "Workspace", "Audio", "Lighting"].map((pill) => (
                <span key={pill} className="rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-center font-semibold">
                  {pill}
                </span>
              ))}
            </div>
            <div className="mt-6 rounded-2xl bg-white p-4 text-slate-900 shadow-lg">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold text-primary-600">Trending pick</p>
                  <h3 className="text-lg font-bold">LED desk lamp + wireless charger</h3>
                  <p className="text-sm text-slate-600">Bundle price - save 18%</p>
                </div>
                <span className="rounded-full bg-primary-100 px-3 py-1 text-xs font-bold text-primary-700">Fast ship</span>
              </div>
              <div className="mt-3 flex items-center justify-between">
                <div className="text-sm text-slate-600">
                  <div className="font-bold text-slate-900 text-xl">$59.99</div>
                  <div>Est. delivery: 2-4 days</div>
                </div>
                <Link
                  to="/product/4"
                  className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                >
                  View bundle
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-900">Shop by category</h2>
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
                  {option}
                </button>
              ))}
            </div>
            <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
              View all
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
                to="/products"
                className="group relative min-w-[200px] sm:min-w-[220px] rounded-xl border border-slate-200 bg-slate-50 px-4 py-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-primary-300 hover:bg-white"
              >
                <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${category.accent} opacity-70`} />
                <div className="relative">
                  <p className="text-sm font-bold text-slate-800 group-hover:text-primary-700">{category.title}</p>
                  <p className="mt-1 text-xs text-slate-500">{category.blurb}</p>
                  <p className="mt-3 text-xs font-semibold text-primary-600">Explore now</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
        <h2 className="text-lg font-bold text-slate-900 mb-6 text-center">Why shop with AUREX</h2>
        <TrustBadges />
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
        <div className="flex flex-wrap items-start gap-4 justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900">Must-have pages covered</h2>
            <p className="text-sm text-slate-600">Borrowed from Barrel NY checklist - quick access below.</p>
          </div>
          <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">See all products</Link>
        </div>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
          {mustHavePages.map((page) => (
            <Link
              key={page.label}
              to={page.to}
              className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800 hover:border-primary-200 hover:bg-primary-50"
            >
              <span className="h-6 w-6 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs">Go</span>
              <span>{page.label}</span>
              <span className="ml-auto text-xs text-primary-600">Open</span>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-center justify-between gap-4">
          <h2 className="text-xl md:text-2xl font-bold text-slate-900">Recommended for you</h2>
          <Link to="/products" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
            View all products
          </Link>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
          {mockProducts.map((product) => (
            <ProductCard product={product} key={product.id} />
          ))}
        </div>
      </section>
    </div>
  );
}
