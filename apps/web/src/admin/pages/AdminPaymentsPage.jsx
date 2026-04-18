import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";
import { StatusBadge } from "../components/AdminCards";

const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function AdminPaymentsPage() {
  const { payments, orders, updatePayment } = useCommerce();

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Payments</h1>
      <p style={{ textAlign: "center", color: "#64748b", marginBottom: 20 }}>Stripe-first payment records with room for additional providers.</p>
      <AdminTable
        rows={payments}
        getRowKey={(payment) => payment.id}
        columns={[
          { key: "providerTransactionId", label: "Transaction" },
          { key: "orderId", label: "Order", render: (payment) => orders.find((order) => order.id === payment.orderId)?.orderNumber || payment.orderId },
          { key: "provider", label: "Provider" },
          { key: "status", label: "Status", render: (payment) => <StatusBadge status={payment.status} /> },
          { key: "amount", label: "Amount", render: (payment) => currency.format(payment.amount) },
          { key: "createdAt", label: "Created", render: (payment) => new Date(payment.createdAt).toLocaleString() },
          {
            key: "actions",
            label: "Actions",
            render: (payment) => (
              <div style={{ display: "flex", gap: 8 }}>
                <button className="aurex-button" onClick={() => updatePayment(payment.id, { status: "succeeded" })}>Capture</button>
                <button className="aurex-button" onClick={() => updatePayment(payment.id, { status: "refunded" })}>Refund</button>
                <button className="aurex-button aurex-button-danger" onClick={() => updatePayment(payment.id, { status: "failed" })}>Fail</button>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}
