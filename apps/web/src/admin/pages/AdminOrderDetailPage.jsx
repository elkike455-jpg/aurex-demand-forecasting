import { Link, useParams } from "react-router-dom";
import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";
import { StatusBadge } from "../components/AdminCards";

const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function AdminOrderDetailPage() {
  const { id } = useParams();
  const { orders, updateOrder } = useCommerce();
  const order = orders.find((item) => item.id === id);

  if (!order) return <div>Order not found</div>;

  return (
    <div className="space-y-6">
      <Link to="/admin/orders" className="text-sm font-bold text-cyan-700">Back to orders</Link>
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-black">{order.orderNumber}</h1>
            <p className="text-slate-500">{order.customerEmail}</p>
          </div>
          <div className="flex gap-2">
            <StatusBadge status={order.status} />
            <StatusBadge status={order.paymentStatus} />
          </div>
        </div>
        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          <Info label="Subtotal" value={currency.format(order.subtotal)} />
          <Info label="Shipping" value={currency.format(order.shipping)} />
          <Info label="Total" value={currency.format(order.total)} />
        </div>
      </div>
      <AdminTable
        rows={order.items}
        getRowKey={(item) => `${item.productId}-${item.sku}`}
        columns={[
          { key: "sku", label: "SKU" },
          { key: "name", label: "Item" },
          { key: "quantity", label: "Qty" },
          { key: "price", label: "Price", render: (item) => currency.format(item.price) },
        ]}
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-lg font-black">Order timeline</h2>
          <div className="mt-4 space-y-3">
            {order.timeline.map((event) => (
              <div key={`${event.status}-${event.at}`} className="border-l-2 border-cyan-300 pl-3">
                <div className="font-bold capitalize">{event.status}</div>
                <div className="text-sm text-slate-500">{new Date(event.at).toLocaleString()} - {event.note}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-lg font-black">Admin action</h2>
          <select value={order.status} onChange={(event) => updateOrder(order.id, { status: event.target.value }, "Status changed from order detail.")} className="mt-4 w-full rounded-lg border-2 border-slate-200 px-3 py-2">
            {["pending", "paid", "processing", "shipped", "delivered", "cancelled", "refunded"].map((status) => <option key={status} value={status}>{status}</option>)}
          </select>
        </div>
      </div>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <div className="text-xs font-black uppercase tracking-[0.14em] text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-black">{value}</div>
    </div>
  );
}
