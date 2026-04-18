import { useMemo, useState } from "react";
import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";
import { StatusBadge } from "../components/AdminCards";

export function AdminUsersPage() {
  const { users, orders } = useCommerce();
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => users.filter((user) => `${user.fullName} ${user.email}`.toLowerCase().includes(query.toLowerCase())),
    [users, query]
  );

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Users</h1>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 24 }}>
        <input className="aurex-input" style={{ maxWidth: 520 }} placeholder="Search Users" value={query} onChange={(event) => setQuery(event.target.value)} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 18, alignItems: "center" }}>
        <h3 style={{ color: "#1976d2", fontWeight: 900 }}>Add new user</h3>
        <button className="aurex-button">Add ↗</button>
      </div>
      <AdminTable
        rows={filtered}
        getRowKey={(user) => user.id}
        columns={[
          { key: "fullName", label: "Name" },
          { key: "phone", label: "Phone Number", render: () => "7020409953" },
          { key: "email", label: "Email" },
          { key: "role", label: "Role", render: (user) => <StatusBadge status={user.role} /> },
          { key: "orders", label: "Orders", render: (user) => orders.filter((order) => order.customerId === user.id).length },
          { key: "createdAt", label: "Created On", render: (user) => new Date(user.createdAt).toLocaleString() },
        ]}
      />
    </div>
  );
}
