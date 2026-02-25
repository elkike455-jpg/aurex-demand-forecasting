import { Link } from "react-router-dom";
import { mockProducts } from "../../mocks/products";
import { ProductCard } from "../components/ProductCard";
import { TrustBadges } from "../components/TrustBadges";

const categories = [
  "Furniture",
  "Lighting & décor",
  "Smart home",
  "Workspace",
  "Audio & speakers",
  "Gaming setup",
  "Kitchen & bar",
  "Garden & outdoor",
];

export function HomePage() {
  return (
    <div className="space-y-10">
      {/* Hero + sidebar layout */}
      <section className="grid gap-6 lg:grid-cols-[260px,minmax(0,1.7fr),minmax(0,1fr)]">
        {/* Categories sidebar */}
        <aside className="rounded-2xl border-2 border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="bg-slate-950 text-slate-50 px-5 py-4 text-base font-bold flex items-center gap-3">
            <span className="h-8 w-8 rounded-full bg-gradient-to-br from-primary-400 to-sky-500 flex items-center justify-center text-sm">
              ☰
            </span>
            <span>Shop by categories</span>
          </div>
          <ul className="divide-y divide-slate-100 text-base">
            {categories.map((category) => (
              <li
                key={category}
                className="flex items-center justify-between px-5 py-3.5 hover:bg-slate-50 cursor-pointer"
              >
                <span className="font-medium text-slate-700">{category}</span>
                <span className="text-slate-400 text-lg">›</span>
              </li>
            ))}
          </ul>
        </aside>

        {/* Main hero card */}
        <div className="relative rounded-2xl overflow-hidden bg-slate-950 text-slate-50 p-8 md:p-10 flex flex-col justify-between shadow-md">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(168,85,247,0.35),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(56,189,248,0.25),_transparent_55%)] pointer-events-none" />

          <div className="relative flex-1 flex flex-col justify-between gap-8">
            <div className="space-y-4 max-w-lg">
              <span className="inline-flex items-center gap-2 rounded-full bg-slate-900/70 px-4 py-2 text-sm font-semibold text-primary-100 ring-1 ring-primary-500/40">
                <span className="h-2 w-2 rounded-full bg-primary-400 animate-pulse" />
                AUREX / New collection
              </span>
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold leading-tight">
                Fresh drops for your{" "}
                <span className="bg-gradient-to-r from-primary-300 via-primary-400 to-sky-300 bg-clip-text text-transparent">
                  future-ready space
                </span>
              </h1>
              <p className="text-base md:text-lg text-slate-200/90 max-w-md leading-relaxed">
                Curated furniture, lighting and tech to make your home feel like
                tomorrow. Hand-picked from the best brands on AUREX.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-5">
              <button className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-8 py-3.5 text-base font-bold text-white shadow-md hover:from-primary-400 hover:to-sky-400 transition-colors">
                Shop now
              </button>
              <button className="rounded-full border-2 border-slate-600/70 bg-slate-900/40 px-6 py-3 text-sm font-semibold text-slate-100 hover:border-primary-400 hover:text-primary-100 transition-colors">
                View collections
              </button>
              <div className="ml-auto text-right text-sm text-slate-300 space-y-1">
                <p>
                  <span className="font-bold text-primary-200 text-base">
                    2.3k+
                  </span>{" "}
                  products in stock
                </p>
                <p className="font-medium">Ships in 24–48 hours</p>
              </div>
            </div>
          </div>
        </div>

        {/* Side banners */}
        <div className="space-y-5">
          <div className="rounded-2xl border-2 border-slate-200 bg-gradient-to-br from-slate-900 via-slate-950 to-primary-700 text-slate-50 p-6 flex flex-col justify-between shadow-md min-h-[160px]">
            <div>
              <p className="text-sm uppercase tracking-wider font-semibold text-primary-100/90 mb-2">
                Limited offer
              </p>
              <h3 className="text-lg font-bold">Smart lighting bundles</h3>
              <p className="mt-2 text-sm text-slate-200/90">
                Up to{" "}
                <span className="font-bold text-primary-100">30%</span> off
                selected sets.
              </p>
            </div>
            <button className="mt-4 self-start rounded-full border-2 border-primary-300/60 bg-slate-950/40 px-4 py-2 text-sm font-semibold text-primary-50 hover:bg-primary-500/20 transition-colors">
              Explore deals
            </button>
          </div>

          <div className="rounded-2xl bg-white border-2 border-slate-200 shadow-sm p-6 flex flex-col justify-between min-h-[160px]">
            <div>
              <p className="text-sm uppercase tracking-wider font-semibold text-slate-500 mb-2">
                Today&apos;s best deals
              </p>
              <h3 className="text-lg font-bold text-slate-900">
                Ends in 31:04:57
              </h3>
              <p className="mt-2 text-sm text-slate-600">
                Discover time-limited offers on top-rated AUREX picks.
              </p>
            </div>
            <button className="mt-4 self-start rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 transition-colors">
              View all deals
            </button>
          </div>
        </div>
      </section>

      {/* Trust – build credibility (ecommerce best practices) */}
      <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
        <h2 className="text-lg font-bold text-slate-900 mb-6 text-center">Why shop with AUREX</h2>
        <TrustBadges />
      </section>

      {/* Products grid */}
      <section>
        <div className="mb-5 flex items-center justify-between gap-4">
          <h2 className="text-xl md:text-2xl font-bold text-slate-900">
            Recommended for you
          </h2>
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

