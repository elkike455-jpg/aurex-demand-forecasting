import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useCart } from "../context/CartContext";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";

export function CheckoutPage() {
  const { items, totalPrice, totalItems, clearCart } = useCart();
  const navigate = useNavigate();
  const [paymentMethod, setPaymentMethod] = useState("card");
  const [form, setForm] = useState({
    email: "",
    fullName: "",
    address: "",
    city: "",
    zip: "",
    card: "",
    exp: "",
    cvv: "",
  });
  const [placed, setPlaced] = useState(false);

  const shipping = totalPrice >= 99 ? 0 : 9.99;
  const total = totalPrice + shipping;

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    clearCart();
    setPlaced(true);
    setTimeout(() => navigate("/"), 3000);
  };

  if (items.length === 0 && !placed) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Your cart is empty</h2>
        <Link to="/products" className="text-primary-600 font-semibold hover:underline">
          Add items to checkout
        </Link>
      </div>
    );
  }

  if (placed) {
    return (
      <div className="text-center py-16 max-w-md mx-auto">
        <div className="text-6xl mb-4">OK</div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Order placed!</h2>
        <p className="text-slate-600 mb-6">
          Thank you for your order. You will receive a confirmation email shortly.
        </p>
        <Link
          to="/"
          className="inline-block rounded-full bg-primary-600 px-8 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
        >
          Back to home
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <Breadcrumbs items={[{ label: "Cart", to: "/cart" }, { label: "Checkout" }]} />
      <h1 className="text-2xl md:text-3xl font-bold text-slate-900">Checkout</h1>

      <p className="rounded-xl bg-primary-50 border border-primary-200 px-4 py-3 text-sm font-medium text-primary-800">
        No account needed - continue as guest. We&apos;ll send your receipt to your email.
      </p>

      <form onSubmit={handleSubmit} className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-6">
          <section className="rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">Contact and shipping</h2>
            <div className="space-y-4">
              <input
                type="email"
                name="email"
                placeholder="Email"
                value={form.email}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <input
                type="text"
                name="fullName"
                placeholder="Full name"
                value={form.fullName}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <input
                type="text"
                name="address"
                placeholder="Address"
                value={form.address}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <div className="grid grid-cols-2 gap-4">
                <input
                  type="text"
                  name="city"
                  placeholder="City"
                  value={form.city}
                  onChange={handleChange}
                  required
                  className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                />
                <input
                  type="text"
                  name="zip"
                  placeholder="ZIP"
                  value={form.zip}
                  onChange={handleChange}
                  required
                  className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                />
              </div>
            </div>
          </section>

          <section className="rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">Payment</h2>
            <div className="flex gap-4 mb-4">
              <label className="flex-1 cursor-pointer">
                <input
                  type="radio"
                  name="paymentMethod"
                  checked={paymentMethod === "card"}
                  onChange={() => setPaymentMethod("card")}
                  className="sr-only peer"
                />
                <span className="block rounded-lg border-2 border-slate-200 px-4 py-3 text-center font-medium peer-checked:border-primary-500 peer-checked:bg-primary-50 peer-checked:text-primary-700">
                  Credit / Debit card
                </span>
              </label>
              <label className="flex-1 cursor-pointer">
                <input
                  type="radio"
                  name="paymentMethod"
                  checked={paymentMethod === "paypal"}
                  onChange={() => setPaymentMethod("paypal")}
                  className="sr-only peer"
                />
                <span className="block rounded-lg border-2 border-slate-200 px-4 py-3 text-center font-medium text-slate-700 peer-checked:border-primary-500 peer-checked:bg-primary-50 peer-checked:text-primary-700">
                  PayPal
                </span>
              </label>
            </div>
            {paymentMethod === "card" && (
              <div className="space-y-4">
                <input
                  type="text"
                  name="card"
                  placeholder="Card number"
                  value={form.card}
                  onChange={handleChange}
                  required={paymentMethod === "card"}
                  className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                />
                <div className="grid grid-cols-2 gap-4">
                  <input
                    type="text"
                    name="exp"
                    placeholder="MM/YY"
                    value={form.exp}
                    onChange={handleChange}
                    required={paymentMethod === "card"}
                    className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                  />
                  <input
                    type="text"
                    name="cvv"
                    placeholder="CVV"
                    value={form.cvv}
                    onChange={handleChange}
                    required={paymentMethod === "card"}
                    className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                  />
                </div>
              </div>
            )}
            {paymentMethod === "paypal" && (
              <p className="text-sm text-slate-600 py-2">
                You&apos;ll be redirected to PayPal to complete payment securely.
              </p>
            )}
          </section>
        </div>

        <div>
          <div className="sticky top-24 rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">Order summary</h2>
            <p className="text-slate-600 mb-4">
              {totalItems} {totalItems === 1 ? "item" : "items"}
            </p>
            <div className="space-y-2 text-slate-600">
              <div className="flex justify-between">
                <span>Subtotal</span>
                <span>${totalPrice.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Shipping</span>
                <span>{shipping === 0 ? "Free" : `$${shipping.toFixed(2)}`}</span>
              </div>
            </div>
            <div className="border-t border-slate-200 my-4" />
            <div className="flex justify-between text-lg font-bold text-slate-900 mb-6">
              <span>Total</span>
              <span>${total.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-600 mb-4">
              <span aria-hidden>Secure</span>
              <span>Secure checkout. Your data is encrypted.</span>
            </div>
            <button
              type="submit"
              className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition shadow-md hover:shadow-lg"
            >
              Place order
            </button>
            <Link
              to="/cart"
              className="mt-4 block text-center text-sm font-medium text-primary-600 hover:underline"
            >
              &lt;- Back to cart
            </Link>
          </div>
        </div>
      </form>

      <div className="pt-6 border-t border-slate-200">
        <TrustBadges compact />
      </div>
    </div>
  );
}
