import { useAuth } from "../../context/AuthContext";
import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";

export function AdminVendorsPage() {
  const { user } = useAuth();
  const { sellers, sellerApplications, categories, products, reviewSellerApplication } = useCommerce();
  const categoryName = (id) => categories.find((category) => category.id === id)?.name || id;

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Providers</h1>
      <div className="aurex-dashboard-grid" style={{ marginBottom: 24 }}>
        <div className="aurex-stat-card aurex-stat-blue">
          <span>Approved providers</span>
          <strong>{sellers.length}</strong>
        </div>
        <div className="aurex-stat-card aurex-stat-pink">
          <span>Pending review</span>
          <strong>{sellerApplications.filter((item) => item.status === "pending").length}</strong>
        </div>
        <div className="aurex-stat-card aurex-stat-yellow">
          <span>Provider products</span>
          <strong>{products.filter((product) => product.sellerId).length}</strong>
        </div>
        <div className="aurex-stat-card aurex-stat-purple">
          <span>Admin profile</span>
          <strong>{user?.role === "superadmin" ? "Root" : "Admin"}</strong>
        </div>
      </div>

      <section style={{ marginBottom: 28 }}>
        <h2 style={{ color: "#1976d2", fontWeight: 900, marginBottom: 12 }}>Seller applications</h2>
        <AdminTable
          rows={sellerApplications}
          getRowKey={(application) => application.id}
          columns={[
            { key: "companyName", label: "Company" },
            { key: "applicantEmail", label: "Email" },
            { key: "taxId", label: "Tax ID" },
            { key: "categories", label: "Categories", render: (application) => application.categories.map(categoryName).join(", ") },
            { key: "status", label: "Status" },
            {
              key: "actions",
              label: "Actions",
              render: (application) => (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button className="aurex-button" type="button" onClick={() => reviewSellerApplication({ applicationId: application.id, status: "approved", reviewer: user })}>Approve</button>
                  <button className="aurex-button aurex-button-danger" type="button" onClick={() => reviewSellerApplication({ applicationId: application.id, status: "rejected", reviewer: user })}>Reject</button>
                </div>
              ),
            },
          ]}
        />
      </section>

      <section>
        <h2 style={{ color: "#1976d2", fontWeight: 900, marginBottom: 12 }}>Approved providers</h2>
        <AdminTable
          rows={sellers}
          getRowKey={(seller) => seller.id}
          columns={[
            { key: "publicName", label: "Seller" },
            { key: "legalName", label: "Legal Name" },
            { key: "taxId", label: "Tax ID" },
            { key: "riskStatus", label: "Risk" },
            { key: "products", label: "Products", render: (seller) => products.filter((product) => product.sellerId === seller.id).length },
            { key: "description", label: "What they sell", render: (seller) => seller.productDescription },
          ]}
        />
      </section>
    </div>
  );
}
