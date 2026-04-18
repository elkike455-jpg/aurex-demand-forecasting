import { useState } from "react";
import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";
import { ForecastHookCard } from "../components/AdminCards";

export function AdminInventoryPage() {
  const { products, inventoryMovements, adjustInventory } = useCommerce();
  const [selectedProductId, setSelectedProductId] = useState(products[0]?.id || "");
  const [quantity, setQuantity] = useState(1);
  const lowStock = products.filter((product) => product.stock <= product.reorderPoint);

  const submit = (event) => {
    event.preventDefault();
    if (!selectedProductId || !quantity) return;
    adjustInventory(selectedProductId, Number(quantity), "Manual inventory update from admin");
    setQuantity(1);
  };

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Inventory</h1>

      <form className="aurex-form-card" onSubmit={submit}>
        <h3 style={{ color: "#1976d2", fontWeight: 900, marginBottom: 14 }}>Update real stock</h3>
        <div className="aurex-form-grid">
          <select className="aurex-input" value={selectedProductId} onChange={(event) => setSelectedProductId(event.target.value)}>
            {products.map((product) => <option key={product.id} value={product.id}>{product.sku} - {product.name}</option>)}
          </select>
          <input className="aurex-input" type="number" value={quantity} onChange={(event) => setQuantity(event.target.value)} placeholder="Quantity (+ restock, - adjust)" />
          <button className="aurex-button">Apply Stock Change</button>
        </div>
      </form>

      <h3 className="aurex-chart-title" style={{ color: "#f59e0b" }}>Low Stock Warnings</h3>
      <AdminTable
        rows={lowStock}
        getRowKey={(product) => product.id}
        columns={[
          { key: "sku", label: "SKU" },
          { key: "name", label: "Product" },
          { key: "stock", label: "Stock" },
          { key: "reorderPoint", label: "Reorder Point" },
          { key: "action", label: "Action", render: (product) => <button className="aurex-button" onClick={() => adjustInventory(product.id, product.reorderPoint + 5, "Restock from low-stock table")}>Restock</button> },
        ]}
      />

      <h3 className="aurex-chart-title" style={{ color: "#1976d2" }}>Inventory Movements</h3>
      <AdminTable
        rows={inventoryMovements}
        getRowKey={(movement) => movement.id}
        columns={[
          { key: "productId", label: "Product ID" },
          { key: "type", label: "Type" },
          { key: "quantity", label: "Quantity" },
          { key: "reason", label: "Reason" },
          { key: "createdAt", label: "Created", render: (movement) => new Date(movement.createdAt).toLocaleString() },
        ]}
      />

      <ForecastHookCard title="Forecast-ready inventory feed" />
    </div>
  );
}
