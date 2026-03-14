import { useState } from "react";
import { Header } from "../components/Header";
import { TrustBadges } from "../components/TrustBadges";
import { MiniCartDrawer } from "../components/MiniCartDrawer";
import { Link } from "react-router-dom";

export function AppLayout({ children }) {
  const [isCartOpen, setIsCartOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <Header onCartOpen={() => setIsCartOpen(true)} />
      <main className="max-w-6xl mx-auto w-full px-5 py-8 flex-1">{children}</main>
      <MiniCartDrawer open={isCartOpen} onClose={() => setIsCartOpen(false)} />
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
                <li><Link to="/gift-cards" className="text-slate-600 hover:text-primary-600">Gift cards</Link></li>
                <li><Link to="/stores" className="text-slate-600 hover:text-primary-600">Store locator</Link></li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">Support</div>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><Link to="/contact" className="hover:text-primary-600">Contact</Link></li>
                <li><Link to="/help" className="hover:text-primary-600">FAQ</Link></li>
                <li><Link to="/shipping-returns" className="hover:text-primary-600">Shipping & returns</Link></li>
                <li><Link to="/track-order" className="hover:text-primary-600">Track order</Link></li>
              </ul>
            </div>
            <div>
              <div className="font-bold text-slate-900 mb-3">Legal</div>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><Link to="/privacy" className="hover:text-primary-600">Privacy</Link></li>
                <li><Link to="/terms" className="hover:text-primary-600">Terms</Link></li>
                <li><Link to="/about" className="hover:text-primary-600">About</Link></li>
                <li><Link to="/careers" className="hover:text-primary-600">Careers</Link></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-top border-slate-200 flex flex-wrap items-center justify-between gap-4 text-sm text-slate-600">
            <span>© {new Date().getFullYear()} AUREX. All rights reserved.</span>
            <TrustBadges compact />
          </div>
        </div>
      </footer>
    </div>
  );
}
