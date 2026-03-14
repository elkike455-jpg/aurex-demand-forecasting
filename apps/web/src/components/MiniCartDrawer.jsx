import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";

export function MiniCartDrawer({ open, onClose }) {
  const { items, removeItem, updateQuantity, totalPrice } = useCart();

  return (
    <div
      className={`${open ? "visible opacity-100" : "invisible opacity-0"} fixed inset-0 z-30 transition-opacity duration-200`}
    >
      <div
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
        onClick={onClose}
      />
      <aside
        className={`absolute right-0 top-0 h-full w-full max-w-md bg-white shadow-2xl border-l border-slate-100 transform transition-transform duration-200 ${open ? "translate-x-0" : "translate-x-full"}`}
        aria-label="Mini cart"
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Your cart</h2>
            <p className="text-sm text-slate-500">Fast preview, checkout anytime</p>
          </div>
          <button
            onClick={onClose}
            className="h-9 w-9 rounded-full border border-slate-200 text-slate-500 hover:text-slate-800 hover:border-slate-300"
            aria-label="Close cart"
          >
            ×
          </button>
        </div>

        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {items.length === 0 && (
              <div className="text-center text-slate-600 py-8">
                Your cart is empty.
              </div>
            )}
            {items.map(({ product, quantity }) => (
              <div key={product.id} className="flex gap-4 rounded-xl border border-slate-100 p-3 shadow-sm">
                <div className="h-16 w-16 rounded-lg bg-slate-100 overflow-hidden">
                  <img src={product.image} alt={product.name} className="h-full w-full object-cover" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-slate-900 truncate">{product.name}</p>
                  <p className="text-sm text-slate-500">${product.price.toFixed(2)}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <button
                      onClick={() => updateQuantity(product.id, quantity - 1)}
                      className="h-8 w-8 rounded-lg border border-slate-200 text-sm font-bold text-slate-700 hover:bg-slate-50"
                    >
                      -
                    </button>
                    <span className="w-8 text-center text-sm font-semibold">{quantity}</span>
                    <button
                      onClick={() => updateQuantity(product.id, quantity + 1)}
                      className="h-8 w-8 rounded-lg border border-slate-200 text-sm font-bold text-slate-700 hover:bg-slate-50"
                    >
                      +
                    </button>
                    <button
                      onClick={() => removeItem(product.id)}
                      className="ml-auto text-sm font-medium text-red-500 hover:text-red-600"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-slate-100 p-6 space-y-3">
            <div className="flex justify-between text-slate-700">
              <span>Subtotal</span>
              <span className="font-semibold text-slate-900">${totalPrice.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <span aria-hidden>??</span>
              <span>Secure checkout with encrypted payments</span>
            </div>
            <div className="flex gap-3 pt-2">
              <Link
                to="/cart"
                onClick={onClose}
                className="flex-1 rounded-full border-2 border-slate-200 px-4 py-3 text-sm font-semibold text-slate-800 hover:border-primary-400 hover:text-primary-600 text-center"
              >
                View cart
              </Link>
              <Link
                to="/checkout"
                onClick={onClose}
                className="flex-1 rounded-full bg-gradient-to-r from-primary-500 to-sky-500 px-4 py-3 text-sm font-bold text-white text-center shadow-md hover:from-primary-400 hover:to-sky-400"
              >
                Checkout
              </Link>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
