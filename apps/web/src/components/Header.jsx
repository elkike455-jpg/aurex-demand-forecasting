import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";
import { useAuth } from "../context/AuthContext";

const QUICK_LINKS = [
  { label: "Home", to: "/" },
  { label: "Shop all", to: "/products" },
  { label: "New arrivals", to: "/products" },
  { label: "Deals", to: "/products" },
  { label: "Gift cards", to: "/gift-cards" },
  { label: "Shipping & returns", to: "/shipping-returns" },
  { label: "Help / FAQ", to: "/help" },
  { label: "Track order", to: "/track-order" },
  { label: "Stores", to: "/stores" },
];

export function Header({ onCartOpen = () => {} }) {
  const { totalItems } = useCart();
  const { user, isAuthenticated, signOut } = useAuth();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);

  const suggestions = useMemo(() => {
    const base = [
      "Track my order",
      "Sofas and sectionals",
      "Smart lighting bundles",
      "Wireless audio",
      "Workspace setup",
      "Shipping & returns",
      "Gift cards",
    ];
    if (!query) return base.slice(0, 5);
    return base
      .filter((item) => item.toLowerCase().includes(query.toLowerCase()))
      .slice(0, 5);
  }, [query]);

  return (
    <header className="sticky top-0 z-20 border-b bg-white/90 backdrop-blur">
      <div className="hidden md:block border-b bg-slate-950 text-sm text-slate-200">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-5 py-2.5">
          <div className="flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Free shipping $99+, 30-day returns, secure checkout</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Live chat</span>
            <span className="h-1.5 w-1.5 rounded-full bg-primary-400" />
            <span>Support 24/7</span>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-5 py-4 flex items-center gap-5">
        <Link to="/" className="flex items-center gap-3 cursor-pointer">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-primary-400 via-primary-500 to-sky-500 flex items-center justify-center text-xl font-black text-white shadow-md">
            A
          </div>
          <div>
            <div className="font-black text-2xl tracking-tight text-slate-900">AUREX</div>
            <div className="text-xs uppercase tracking-[0.16em] text-slate-500 font-semibold">
              Futuristic marketplace
            </div>
          </div>
        </Link>

        <div className="flex-1 max-w-xl relative">
          <div className="flex items-center gap-2 rounded-full border-2 border-slate-200 bg-slate-50 px-5 py-3 shadow-sm focus-within:border-primary-400 focus-within:bg-white transition-colors">
            <input
              type="text"
              placeholder="Search products, brands and more..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setTimeout(() => setFocused(false), 120)}
              className="w-full bg-transparent outline-none text-base placeholder:text-slate-400"
            />
            <button className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-5 py-2 text-base font-semibold text-white shadow-sm hover:from-primary-400 hover:to-sky-400 transition-colors">
              Search
            </button>
          </div>
          {focused && suggestions.length > 0 && (
            <div className="absolute left-0 right-0 mt-2 rounded-2xl border border-slate-100 bg-white shadow-xl p-3 text-sm text-slate-700">
              <div className="font-semibold text-slate-900 px-2 pb-2">Popular searches</div>
              <ul className="divide-y divide-slate-100">
                {suggestions.map((s) => (
                  <li key={s} className="px-2 py-2 hover:bg-slate-50 rounded-xl cursor-pointer">
                    {s}
                  </li>
                ))}
              </ul>
              <div className="flex items-center gap-2 px-2 pt-2 text-xs text-slate-500">
                <span aria-hidden>Top</span>
                <span>Try "sofa", "wireless", "desk lamp"</span>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 text-base">
          {isAuthenticated ? (
            <>
              <div className="hidden md:block text-right leading-tight">
                <p className="text-xs uppercase tracking-[0.12em] text-slate-500 font-semibold">Account</p>
                <p className="font-semibold text-slate-800">{user.fullName}</p>
              </div>
              <button
                onClick={signOut}
                type="button"
                className="rounded-full border-2 border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                Sign out
              </button>
            </>
          ) : (
            <>
              <Link to="/sign-in" className="font-medium text-slate-700 hover:text-primary-600">
                Sign in
              </Link>
              <Link
                to="/create-account"
                className="hidden sm:inline-flex rounded-full border-2 border-slate-300 px-4 py-2 text-sm font-semibold hover:border-primary-500 hover:text-primary-600 transition-colors"
              >
                Create account
              </Link>
            </>
          )}
          <button
            onClick={onCartOpen}
            className="relative p-2 hover:text-primary-600 text-sm font-semibold"
            aria-label="Cart"
            type="button"
          >
            Cart
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
              key={item.label}
              to={item.to}
              className={`whitespace-nowrap font-medium hover:text-primary-600 ${
                item.label === "Home" ? "text-slate-900 font-semibold" : ""
              }`}
            >
              {item.label}
            </Link>
          ))}
          <span className="ml-auto hidden sm:inline-flex items-center gap-2 text-xs uppercase tracking-[0.12em] font-semibold text-slate-500">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            Limited-time bundles
          </span>
        </div>
      </nav>
    </header>
  );
}
