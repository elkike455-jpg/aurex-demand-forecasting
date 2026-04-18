import { useState } from "react";
import { Link } from "react-router-dom";
import { useCommerce } from "../../context/CommerceContext";
import { StatusBadge } from "../components/AdminCards";

const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function AdminOrdersPage() {
  const { orders, updateOrder } = useCommerce();
  const [openOrderId, setOpenOrderId] = useState("");
  const sortedOrders = [...orders].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Orders</h1>
      <div className="aurex-table-card">
        <table className="aurex-table">
          <thead>
            <tr>
              <th />
              <th>User Name</th>
              <th>Email</th>
              <th>Total Amount</th>
              <th>Status</th>
              <th>Order Created Date</th>
            </tr>
          </thead>
          <tbody>
            {sortedOrders.map((order) => (
              <>
                <tr key={order.id}>
                  <td>
                    <button className="aurex-button" type="button" onClick={() => setOpenOrderId(openOrderId === order.id ? "" : order.id)}>
                      {openOrderId === order.id ? "⌃" : "⌄"}
                    </button>
                  </td>
                  <td><Link to={`/admin/orders/${order.id}`}>{order.customerEmail.split("@")[0]}</Link></td>
                  <td><Link to={`/admin/orders/${order.id}`}>{order.customerEmail}</Link></td>
                  <td>{currency.format(order.total)}</td>
                  <td>
                    <select value={order.status} onChange={(event) => updateOrder(order.id, { status: event.target.value }, "Status changed in admin.")} className="aurex-input" style={{ width: 140 }}>
                      {["pending", "paid", "processing", "shipped", "delivered", "cancelled", "refunded"].map((status) => <option key={status} value={status}>{status}</option>)}
                    </select>
                  </td>
                  <td>{new Date(order.createdAt).toLocaleString()}</td>
                </tr>
                {openOrderId === order.id && (
                  <tr>
                    <td colSpan="6">
                      <div style={{ padding: 10 }}>
                        <p>Payment: <StatusBadge status={order.paymentStatus} /></p>
                        <p>Admin Notes: {order.notes || "No notes yet"}</p>
                        <table className="aurex-table" style={{ marginTop: 12 }}>
                          <thead>
                            <tr>
                              <th>Product Name</th>
                              <th>SKU</th>
                              <th>Price</th>
                              <th>Quantity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {order.items.map((item) => (
                              <tr key={`${order.id}-${item.productId}`}>
                                <td>{item.name}</td>
                                <td>{item.sku}</td>
                                <td>{currency.format(item.price)}</td>
                                <td>{item.quantity}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
