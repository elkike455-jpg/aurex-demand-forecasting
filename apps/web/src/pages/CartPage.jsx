import { useState } from "react";
import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";
import { Breadcrumbs } from "../components/Breadcrumbs";

export function CartPage() {
  const [paymentMethod, setPaymentMethod] = useState("card");
  const { items, removeItem, updateQuantity, totalPrice, totalItems } = useCart();
  const total = totalPrice + (totalPrice >= 99 ? 0 : 9.99);

  if (items.length === 0) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Your cart is empty</h2>
        <p className="text-slate-600 mb-6">Add items from the shop to get started.</p>
        <Link
          to="/products"
          className="inline-block rounded-full bg-primary-600 px-8 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
        >
          Browse products
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Shopping cart" }]} />
      <h1 className="text-2xl md:text-3xl font-bold text-slate-900">
        Shopping cart ({totalItems} {totalItems === 1 ? "item" : "items"})
      </h1>

      <div className="grid gap-6 lg:grid-cols-[1fr,320px]">
        <ul className="space-y-4">
          {items.map(({ product, quantity }) => (
            <li
              key={product.id}
              className="flex gap-4 rounded-xl border-2 border-slate-200 bg-white p-4"
            >
              <div className="h-24 w-24 shrink-0 overflow-hidden rounded-lg bg-slate-100">
                <img
                  src={product.image}
                  alt={product.name}
                  className="h-full w-full object-cover"
                />
              </div>
              <div className="flex flex-1 flex-col justify-between min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <Link
                    to={`/product/${product.id}`}
                    className="font-semibold text-slate-900 hover:text-primary-600 hover:underline"
                  >
                    {product.name}
                  </Link>
                  <button
                    onClick={() => removeItem(product.id)}
                    className="text-slate-400 hover:text-red-600 text-sm font-medium"
                    aria-label="Remove"
                  >
                    Remove
                  </button>
                </div>
                <div className="flex items-center justify-between gap-4 mt-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => updateQuantity(product.id, quantity - 1)}
                      className="h-9 w-9 rounded-lg border-2 border-slate-200 font-bold text-slate-600 hover:bg-slate-50"
                    >
                      -
                    </button>
                    <span className="w-10 text-center font-semibold">{quantity}</span>
                    <button
                      onClick={() => updateQuantity(product.id, quantity + 1)}
                      className="h-9 w-9 rounded-lg border-2 border-slate-200 font-bold text-slate-600 hover:bg-slate-50"
                    >
                      +
                    </button>
                  </div>
                  <span className="text-lg font-bold text-slate-900">
                    ${(product.price * quantity).toFixed(2)}
                  </span>
                </div>
              </div>
            </li>
          ))}
        </ul>

        <div className="h-fit rounded-xl border-2 border-slate-200 bg-white p-6">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Order summary</h2>
          <div className="flex justify-between text-slate-600 mb-2">
            <span>Subtotal</span>
            <span>${totalPrice.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-slate-600 mb-2">
            <span>Shipping</span>
            <span>{totalPrice >= 99 ? "Free" : "$9.99"}</span>
          </div>
          <div className="border-t border-slate-200 my-4" />
          <div className="flex justify-between text-lg font-bold text-slate-900 mb-6">
            <span>Total</span>
            <span>${total.toFixed(2)}</span>
          </div>

          <div className="mb-5 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs uppercase tracking-[0.12em] font-semibold text-slate-500 mb-2">Payment method</p>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setPaymentMethod("card")}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold transition-colors ${
                  paymentMethod === "card"
                    ? "border-primary-500 bg-primary-50 text-primary-700"
                    : "border-slate-200 text-slate-600 hover:border-primary-300"
                }`}
              >
                Card
              </button>
              <button
                type="button"
                onClick={() => setPaymentMethod("paypal")}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold transition-colors ${
                  paymentMethod === "paypal"
                    ? "border-primary-500 bg-primary-50 text-primary-700"
                    : "border-slate-200 text-slate-600 hover:border-primary-300"
                }`}
              >
                PayPal
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-500">Secure payment with encrypted processing.</p>
          </div>

          <Link
            to="/checkout"
            className="block w-full rounded-full bg-primary-600 py-3 text-center text-base font-bold text-white hover:bg-primary-700 transition-colors"
          >
            Pay ${total.toFixed(2)} with {paymentMethod === "card" ? "Card" : "PayPal"}
          </Link>
          <p className="mt-3 text-center text-xs text-slate-500">
            You can review your details before placing the final order.
          </p>
        </div>
      </div>
    </div>
  );
}
