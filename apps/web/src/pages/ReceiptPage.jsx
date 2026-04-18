import { Link, useParams } from "react-router-dom";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useCommerce } from "../context/CommerceContext";

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function ReceiptPage() {
  const { id } = useParams();
  const { orders, payments, products, sellers } = useCommerce();
  const order = orders.find((item) => item.id === id);
  const payment = payments.find((item) => item.orderId === id);

  if (!order) {
    return (
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-8 text-center">
        <h1 className="text-2xl font-black text-slate-900">Ticket no encontrado</h1>
        <Link to="/account" className="mt-5 inline-flex rounded-full bg-primary-600 px-6 py-3 font-bold text-white">
          Ver mi cuenta
        </Link>
      </div>
    );
  }

  const sellerName = (productId) => {
    const product = products.find((item) => String(item.id) === String(productId));
    const seller = sellers.find((item) => item.id === product?.sellerId);
    return seller?.publicName || seller?.companyName || "AUREX";
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Breadcrumbs items={[{ label: "Mi cuenta", to: "/account" }, { label: "Ticket de compra" }]} />
      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm print:shadow-none">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 pb-5">
          <div className="flex items-center gap-4">
            <img src="/aurexlogo-transparent.png" alt="AUREX" className="h-16 w-20 object-contain" />
            <div>
              <p className="text-xs font-black uppercase tracking-[0.18em] text-primary-700">Ticket de compra</p>
              <h1 className="text-3xl font-black text-slate-950">{order.orderNumber}</h1>
            </div>
          </div>
          <div className="text-right text-sm text-slate-600">
            <p><b>Fecha:</b> {new Date(order.createdAt).toLocaleString()}</p>
            <p><b>Pago:</b> {payment?.status || order.paymentStatus}</p>
            <p><b>Transaccion:</b> {payment?.providerTransactionId || "simulada"}</p>
          </div>
        </div>

        <div className="grid gap-5 py-5 md:grid-cols-2">
          <div className="rounded-lg bg-slate-50 p-4">
            <h2 className="font-black text-slate-900">Cliente</h2>
            <p className="mt-2 text-sm text-slate-600">{order.shippingAddress?.fullName || order.customerEmail}</p>
            <p className="text-sm text-slate-600">{order.shippingAddress?.email || order.customerEmail}</p>
          </div>
          <div className="rounded-lg bg-slate-50 p-4">
            <h2 className="font-black text-slate-900">Envio</h2>
            <p className="mt-2 text-sm text-slate-600">{order.shippingAddress?.address || "Direccion no capturada"}</p>
            <p className="text-sm text-slate-600">{order.shippingAddress?.city} {order.shippingAddress?.zip}</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-3">Producto</th>
                <th className="py-3">Vendedor</th>
                <th className="py-3 text-right">Cantidad</th>
                <th className="py-3 text-right">Precio</th>
                <th className="py-3 text-right">Total</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {order.items.map((item) => (
                <tr key={`${item.productId}-${item.sku}`}>
                  <td className="py-4 font-bold text-slate-900">{item.name}</td>
                  <td className="py-4 text-slate-600">{sellerName(item.productId)}</td>
                  <td className="py-4 text-right">{item.quantity}</td>
                  <td className="py-4 text-right">{money.format(item.price)}</td>
                  <td className="py-4 text-right font-bold">{money.format(item.price * item.quantity)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="ml-auto mt-6 max-w-sm space-y-2 text-sm">
          <div className="flex justify-between"><span>Subtotal</span><b>{money.format(order.subtotal)}</b></div>
          <div className="flex justify-between"><span>Envio</span><b>{order.shipping === 0 ? "Gratis" : money.format(order.shipping)}</b></div>
          <div className="border-t border-slate-200 pt-3 flex justify-between text-lg text-slate-950"><span>Total</span><b>{money.format(order.total)}</b></div>
        </div>

        <div className="mt-7 flex flex-wrap gap-3 print:hidden">
          <button type="button" onClick={() => window.print()} className="rounded-full bg-primary-600 px-6 py-3 font-black text-white">
            Imprimir ticket
          </button>
          <Link to="/account" className="rounded-full border-2 border-slate-300 px-6 py-3 font-black text-slate-700">
            Ver mi cuenta
          </Link>
        </div>
      </section>
    </div>
  );
}
