import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useCommerce } from "../../context/CommerceContext";
import { useLanguage } from "../../context/LanguageContext";
import { AdminTable } from "../components/AdminTable";

export function AdminProductsPage() {
  const { products, categories, sellers, createProduct, updateProduct, deleteProduct, adjustInventory } = useCommerce();
  const { language } = useLanguage();
  const isEs = language === "es";
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState(defaultProduct(categories[0]?.id));

  const filtered = useMemo(
    () => products.filter((product) => `${product.name} ${product.sku}`.toLowerCase().includes(query.toLowerCase())),
    [products, query]
  );

  const submit = (event) => {
    event.preventDefault();
    if (!form.name || !form.price || !form.categoryId) return;
    editing ? updateProduct(editing.id, form) : createProduct(form);
    setEditing(null);
    setForm(defaultProduct(categories[0]?.id));
  };

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">{isEs ? "Productos" : "Products"}</h1>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 24 }}>
        <input className="aurex-input" style={{ maxWidth: 520 }} placeholder={isEs ? "Buscar productos" : "Search Items"} value={query} onChange={(event) => setQuery(event.target.value)} />
      </div>

      <form className="aurex-form-card" onSubmit={submit}>
        <h3 style={{ color: "#1976d2", fontWeight: 900, marginBottom: 14 }}>
          {editing ? (isEs ? "Editar producto" : "Edit product") : (isEs ? "Agregar producto" : "Add new product")}
        </h3>
        <div className="aurex-form-grid">
          <input className="aurex-input" placeholder={isEs ? "Nombre del producto" : "Product Name"} value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
          <input className="aurex-input" placeholder="SKU" value={form.sku} onChange={(event) => setForm({ ...form, sku: event.target.value })} />
          <input className="aurex-input" placeholder={isEs ? "URL de imagen" : "Image URL"} value={form.image} onChange={(event) => setForm({ ...form, image: event.target.value })} />
          <select className="aurex-input" value={form.categoryId} onChange={(event) => setForm({ ...form, categoryId: event.target.value })}>
            {categories.map((category) => <option key={category.id} value={category.id}>{category.name}</option>)}
          </select>
          <select className="aurex-input" value={form.sellerId} onChange={(event) => setForm({ ...form, sellerId: event.target.value })}>
            {sellers.map((seller) => <option key={seller.id} value={seller.id}>{seller.publicName || seller.companyName}</option>)}
          </select>
          <input className="aurex-input" type="number" placeholder={isEs ? "Precio" : "Price"} value={form.price} onChange={(event) => setForm({ ...form, price: event.target.value })} />
          <input className="aurex-input" type="number" placeholder="Stock" value={form.stock} onChange={(event) => setForm({ ...form, stock: event.target.value })} />
          <input className="aurex-input" type="number" placeholder={isEs ? "Punto de reorden" : "Reorder Point"} value={form.reorderPoint} onChange={(event) => setForm({ ...form, reorderPoint: event.target.value })} />
          <button className="aurex-button">{editing ? (isEs ? "Guardar" : "Save") : (isEs ? "Agregar" : "Add")}</button>
        </div>
        <textarea className="aurex-input" style={{ marginTop: 12 }} placeholder={isEs ? "Descripcion" : "Description"} value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} />
      </form>

      <AdminTable
        rows={filtered}
        getRowKey={(product) => product.id}
        columns={[
          { key: "name", label: isEs ? "Producto" : "Product Name", render: (product) => <Link to={`/admin/products/${product.id}`}>{product.name}</Link> },
          { key: "image", label: isEs ? "Imagen" : "Image", render: (product) => <img src={product.image} alt={product.name} style={{ width: 78, height: 78, objectFit: "contain" }} /> },
          { key: "price", label: isEs ? "Precio" : "Price", render: (product) => `$${Number(product.price).toFixed(2)}` },
          { key: "seller", label: isEs ? "Vendedor" : "Seller", render: (product) => sellers.find((seller) => seller.id === product.sellerId)?.publicName || "AUREX" },
          { key: "stock", label: "Stock", render: (product) => <b style={{ color: product.stock <= product.reorderPoint ? "#f59e0b" : "#111827" }}>{product.stock}</b> },
          { key: "rating", label: isEs ? "Calificacion" : "Rating" },
          {
            key: "actions",
            label: isEs ? "Acciones" : "Actions",
            render: (product) => (
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button className="aurex-button" type="button" onClick={() => { setEditing(product); setForm(product); }}>{isEs ? "Editar" : "Edit"}</button>
                <button className="aurex-button" type="button" onClick={() => updateProduct(product.id, { active: !product.active })}>{product.active ? (isEs ? "Ocultar" : "Hide") : (isEs ? "Mostrar" : "Show")}</button>
                <button className="aurex-button" type="button" onClick={() => adjustInventory(product.id, 5, "Quick restock from product table")}>+5 Stock</button>
                <button className="aurex-button aurex-button-danger" type="button" onClick={() => window.confirm(isEs ? "Eliminar producto?" : "Delete product?") && deleteProduct(product.id)}>{isEs ? "Eliminar" : "Delete"}</button>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}

function defaultProduct(categoryId = "") {
  return { name: "", sku: "", image: "", categoryId, sellerId: "seller_aurex", price: "", stock: "", reorderPoint: 5, description: "" };
}
