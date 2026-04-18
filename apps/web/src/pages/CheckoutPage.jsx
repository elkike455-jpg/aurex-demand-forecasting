import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useCart } from "../context/CartContext";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { TrustBadges } from "../components/TrustBadges";
import { useLanguage } from "../context/LanguageContext";
import { useAuth } from "../context/AuthContext";
import { useCommerce } from "../context/CommerceContext";
import { createCheckoutPayment } from "../payments/paymentGateway";
import { emailReceipt } from "../receipts/receiptEmail";

export function CheckoutPage() {
  const { items, totalPrice, totalItems, clearCart } = useCart();
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { user } = useAuth();
  const { createOrder, recordPayment, products, sellers } = useCommerce();
  const [paymentMethod, setPaymentMethod] = useState("card");
  const [form, setForm] = useState({
    email: user?.email || "",
    fullName: user?.fullName || "",
    address: "",
    city: "",
    zip: "",
    card: "",
    exp: "",
    cvv: "",
  });
  const [placed, setPlaced] = useState(false);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const shipping = totalPrice >= 99 ? 0 : 9.99;
  const total = totalPrice + shipping;

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      const order = createOrder({
        user: user || { id: "guest", email: form.email, fullName: form.fullName },
        items,
        shipping,
        paymentStatus: "pending",
        status: "pending",
        shippingAddress: {
          fullName: form.fullName,
          email: form.email,
          address: form.address,
          city: form.city,
          zip: form.zip,
        },
      });
      const payment = await createCheckoutPayment({ order });
      const recordedPayment = recordPayment({
        order,
        provider: payment.provider,
        providerTransactionId: payment.providerTransactionId,
        status: payment.status,
      });
      try {
        await emailReceipt({
          order: { ...order, status: payment.status === "succeeded" ? "paid" : order.status, paymentStatus: payment.status },
          payment: recordedPayment,
          products,
          sellers,
        });
      } catch (receiptError) {
        console.warn("Receipt email could not be sent", receiptError);
      }
      clearCart();
      if (payment.redirectUrl) {
        window.location.href = payment.redirectUrl;
        return;
      }
      setPlaced(true);
      navigate(`/receipt/${order.id}`, { replace: true });
    } catch (checkoutError) {
      setError(checkoutError.message || "Payment failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (items.length === 0 && !placed) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">{t("cart.emptyTitle")}</h2>
        <Link to="/products" className="text-primary-600 font-semibold hover:underline">
          {t("checkout.emptyLink")}
        </Link>
      </div>
    );
  }

  if (placed) {
    return (
      <div className="text-center py-16 max-w-md mx-auto">
        <div className="text-6xl mb-4">OK</div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">{t("checkout.placedTitle")}</h2>
        <p className="text-slate-600 mb-6">
          {t("checkout.placedBlurb")}
        </p>
        <Link
          to="/"
          className="inline-block rounded-full bg-primary-600 px-8 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
        >
          {t("checkout.backHome")}
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <Breadcrumbs items={[{ label: t("header.cart"), to: "/cart" }, { label: t("checkout.title") }]} />
      <h1 className="text-2xl md:text-3xl font-bold text-slate-900">{t("checkout.title")}</h1>

      <p className="rounded-xl bg-primary-50 border border-primary-200 px-4 py-3 text-sm font-medium text-primary-800">
        {t("checkout.guestNotice")}
      </p>

      <form onSubmit={handleSubmit} className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-6">
          <section className="rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">{t("checkout.contactShipping")}</h2>
            <div className="space-y-4">
              <input
                type="email"
                name="email"
                placeholder={t("checkout.email")}
                value={form.email}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <input
                type="text"
                name="fullName"
                placeholder={t("checkout.fullName")}
                value={form.fullName}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <input
                type="text"
                name="address"
                placeholder={t("checkout.address")}
                value={form.address}
                onChange={handleChange}
                required
                className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
              />
              <div className="grid grid-cols-2 gap-4">
                <input
                  type="text"
                  name="city"
                  placeholder={t("checkout.city")}
                  value={form.city}
                  onChange={handleChange}
                  required
                  className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                />
                <input
                  type="text"
                  name="zip"
                  placeholder={t("checkout.zip")}
                  value={form.zip}
                  onChange={handleChange}
                  required
                  className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
                />
              </div>
            </div>
          </section>

          <section className="rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">{t("checkout.payment")}</h2>
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
                  {t("checkout.creditDebit")}
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
                  placeholder={t("checkout.cardNumber")}
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
                {t("checkout.paypalRedirect")}
              </p>
            )}
          </section>
        </div>

        <div>
          <div className="sticky top-24 rounded-xl border-2 border-slate-200 bg-white p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4">{t("checkout.orderSummary")}</h2>
            <p className="text-slate-600 mb-4">
              {totalItems} {t(totalItems === 1 ? "cart.item_one" : "cart.item_other")}
            </p>
            <div className="space-y-2 text-slate-600">
              <div className="flex justify-between">
                <span>{t("cart.subtotal")}</span>
                <span>${totalPrice.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>{t("cart.shipping")}</span>
                <span>{shipping === 0 ? t("cart.free") : `$${shipping.toFixed(2)}`}</span>
              </div>
            </div>
            <div className="border-t border-slate-200 my-4" />
            <div className="flex justify-between text-lg font-bold text-slate-900 mb-6">
              <span>{t("cart.total")}</span>
              <span>${total.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-600 mb-4">
              <span aria-hidden>Secure</span>
              <span>{t("checkout.secure")}</span>
            </div>
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition shadow-md hover:shadow-lg"
            >
              {isSubmitting ? "Processing..." : t("checkout.placeOrder")}
            </button>
            {error && <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm font-bold text-red-700">{error}</p>}
            <Link
              to="/cart"
              className="mt-4 block text-center text-sm font-medium text-primary-600 hover:underline"
            >
              {t("checkout.backCart")}
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

