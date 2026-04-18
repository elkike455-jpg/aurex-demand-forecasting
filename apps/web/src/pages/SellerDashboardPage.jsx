import { Link } from "react-router-dom";
import { useState } from "react";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { ProductCard } from "../components/ProductCard";
import { useAuth } from "../context/AuthContext";
import { useCommerce } from "../context/CommerceContext";

export function SellerDashboardPage() {
  const { user } = useAuth();
  const { categories, products, sellers, sellerApplications, createProduct } = useCommerce();
  const seller = sellers.find((item) => item.id === user?.sellerId || item.userId === user?.id);
  const application = sellerApplications.find((item) => item.userId === user?.id);
  const sellerId = seller?.id || user?.sellerId;
  const myProducts = products.filter((product) => product.sellerId === sellerId);
  const [form, setForm] = useState({
    name: "",
    sku: "",
    image: "",
    categoryId: categories[0]?.id || "",
    price: "",
    stock: "",
    reorderPoint: 5,
    description: "",
  });

  if (!seller && application?.status !== "approved") {
    return (
      <div className="max-w-3xl mx-auto rounded-2xl border-2 border-slate-200 bg-white p-8">
        <Breadcrumbs items={[{ label: "Panel de vendedor" }]} />
        <h1 className="text-3xl font-black text-slate-950">Tu alta esta en revision</h1>
        <p className="mt-2 text-slate-600">
          Estado actual: <span className="font-bold">{application?.status || "sin solicitud"}</span>. Cuando el admin general apruebe tu proveedor, podras publicar productos.
        </p>
        <Link to="/seller/onboarding" className="mt-5 inline-flex rounded-full bg-primary-600 px-6 py-3 font-bold text-white">
          Completar o reenviar alta
        </Link>
      </div>
    );
  }

  const submit = (event) => {
    event.preventDefault();
    if (!form.name || !form.categoryId || !sellerId) return;
    createProduct({ ...form, sellerId, active: true });
    setForm({ name: "", sku: "", image: "", categoryId: categories[0]?.id || "", price: "", stock: "", reorderPoint: 5, description: "" });
  };

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Panel de vendedor" }]} />
      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-black text-slate-950">{seller?.publicName || seller?.companyName}</h1>
            <p className="mt-2 text-slate-600">Publica productos, cuida inventario y prepara tus datos para forecast de demanda.</p>
          </div>
          <span className="rounded-full bg-emerald-50 px-4 py-2 text-sm font-black text-emerald-700">Proveedor aprobado</span>
        </div>
      </section>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h2 className="text-xl font-black text-slate-900">Publicar producto</h2>
        <form onSubmit={submit} className="mt-5 grid gap-4 md:grid-cols-2">
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Nombre del producto" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="SKU" value={form.sku} onChange={(e) => setForm({ ...form, sku: e.target.value })} />
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="URL de imagen" value={form.image} onChange={(e) => setForm({ ...form, image: e.target.value })} required />
          <select className="rounded-lg border-2 border-slate-200 px-4 py-3" value={form.categoryId} onChange={(e) => setForm({ ...form, categoryId: e.target.value })}>
            {categories.map((category) => <option key={category.id} value={category.id}>{category.name}</option>)}
          </select>
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" type="number" min="0" step="0.01" placeholder="Precio" value={form.price} onChange={(e) => setForm({ ...form, price: e.target.value })} required />
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" type="number" min="0" placeholder="Stock" value={form.stock} onChange={(e) => setForm({ ...form, stock: e.target.value })} required />
          <input className="rounded-lg border-2 border-slate-200 px-4 py-3" type="number" min="0" placeholder="Punto de reorden" value={form.reorderPoint} onChange={(e) => setForm({ ...form, reorderPoint: e.target.value })} />
          <button className="rounded-full bg-primary-600 px-6 py-3 font-black text-white">Publicar</button>
          <textarea className="md:col-span-2 rounded-lg border-2 border-slate-200 px-4 py-3" rows={4} placeholder="Descripcion" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
        </form>
      </section>

      <section>
        <h2 className="text-xl font-black text-slate-900 mb-4">Mis productos</h2>
        {myProducts.length === 0 ? (
          <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white p-8 text-center text-slate-500">Todavia no tienes productos publicados.</div>
        ) : (
          <div className="grid grid-cols-2 gap-5 md:grid-cols-3 lg:grid-cols-4">
            {myProducts.map((product) => <ProductCard key={product.id} product={product} />)}
          </div>
        )}
      </section>
    </div>
  );
}
