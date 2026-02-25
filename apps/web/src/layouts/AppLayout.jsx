import { Header } from "../components/Header";
import { TrustBadges } from "../components/TrustBadges";
import { Link } from "react-router-dom";

export function AppLayout({ children }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <Header />
      <main className="max-w-6xl mx-auto w-full px-5 py-8 flex-1">{children}</main>
      <footer className="border-t-2 border-slate-200 bg-white mt-auto">
        <div className="max-w-6xl mx-auto px-5 py-10">
          <div className="grid gap-8 md:grid-cols-4 mb-10">
            <div>
              <div className="font-bold text-lg text-slate-900 mb-3">AUREX</div>
              <p className="text-sm text-slate-600">
                Futuristic marketplace for tech, furniture, and lifestyle. Secure checkout, fast shipping.
              </p>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">Shop</div>
              <ul className="space-y-2 text-sm">
                <li><Link to="/products" className="text-slate-600 hover:text-primary-600">All products</Link></li>
                <li><Link to="/products" className="text-slate-600 hover:text-primary-600">Deals</Link></li>
                <li><Link to="/cart" className="text-slate-600 hover:text-primary-600">Cart</Link></li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">Support</div>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>Contact us</li>
                <li>Returns &amp; refunds</li>
                <li>Shipping info</li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">Trust</div>
              <TrustBadges compact />
            </div>
          </div>
          <div className="pt-8 border-t border-slate-200 flex flex-wrap items-center justify-between gap-4 text-sm text-slate-600">
            <span>© {new Date().getFullYear()} AUREX. All rights reserved.</span>
            <span className="flex items-center gap-2">
              <span aria-hidden>🔒</span> Secure checkout
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
