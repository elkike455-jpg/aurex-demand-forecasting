import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";

export function Header() {
  const { totalItems } = useCart();

  return (
    <header className="sticky top-0 z-20 border-b bg-white/80 backdrop-blur">
      {/* Top info bar */}
      <div className="hidden md:block border-b bg-slate-950 text-sm text-slate-300">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-5 py-2.5">
          <span>Free shipping on orders over $99</span>
          <span className="flex items-center gap-4">
            <span>Support 24/7</span>
            <span className="h-1.5 w-1.5 rounded-full bg-primary-400" />
            <span>Secure checkout</span>
          </span>
        </div>
      </div>

      {/* Main header */}
      <div className="max-w-6xl mx-auto px-5 py-4 flex items-center gap-5">
        {/* Logo / brand */}
        <Link to="/" className="flex items-center gap-3 cursor-pointer">
          <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-primary-400 via-primary-500 to-sky-500 flex items-center justify-center text-xl font-black text-white shadow-md">
            A
          </div>
          <div>
            <div className="font-bold text-2xl tracking-tight text-slate-900">
              AUREX
            </div>
            <div className="text-sm uppercase tracking-[0.12em] text-slate-500 font-medium">
              Futuristic marketplace
            </div>
          </div>
        </Link>

        {/* Search */}
        <div className="flex-1 max-w-xl">
          <div className="flex items-center gap-2 rounded-full border-2 border-slate-200 bg-slate-50 px-5 py-3 shadow-sm focus-within:border-primary-400 focus-within:bg-white transition-colors">
            <input
              type="text"
              placeholder="Search products, brands and more..."
              className="w-full bg-transparent outline-none text-base placeholder:text-slate-400"
            />
            <button className="rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-5 py-2 text-base font-semibold text-white shadow-sm hover:from-primary-400 hover:to-sky-400 transition-colors">
              Search
            </button>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-5 text-base">
          <button className="font-medium text-slate-700 hover:text-primary-600">Sign in</button>
          <button className="hidden sm:inline-flex rounded-full border-2 border-slate-300 px-4 py-2 text-sm font-semibold hover:border-primary-500 hover:text-primary-600 transition-colors">
            Create account
          </button>
          <Link to="/cart" className="relative p-2 hover:text-primary-600 text-xl" aria-label="Cart">
            🛒
            <span className="absolute -right-0.5 -top-0.5 h-6 w-6 rounded-full bg-primary-500 text-sm font-bold text-white flex items-center justify-center shadow-sm">
              {totalItems}
            </span>
          </Link>
        </div>
      </div>

      {/* Navigation */}
      <nav className="border-t bg-white">
        <div className="max-w-6xl mx-auto px-5 py-3 flex gap-6 text-sm sm:text-base text-slate-600 overflow-x-auto">
          <Link to="/" className="font-semibold text-slate-900 hover:text-primary-600 whitespace-nowrap">
            Home
          </Link>
          <Link to="/products" className="font-medium hover:text-primary-600 whitespace-nowrap">
            All products
          </Link>
          <Link to="/products" className="font-medium hover:text-primary-600 whitespace-nowrap">
            New arrivals
          </Link>
          <Link to="/products" className="font-medium hover:text-primary-600 whitespace-nowrap">
            Deals
          </Link>
          <Link to="/products" className="font-medium hover:text-primary-600 whitespace-nowrap">
            Collections
          </Link>
          <span className="font-medium text-slate-400 whitespace-nowrap cursor-default">
            Support
          </span>
        </div>
      </nav>
    </header>
  );
}

