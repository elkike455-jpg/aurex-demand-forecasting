import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useAuth } from "../context/AuthContext";
import { useCommerce } from "../context/CommerceContext";

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function AccountPage() {
  const { user, updateCurrentUser } = useAuth();
  const { users, orders, sellers, sellerApplications, products, categories, updateUser } = useCommerce();
  const storedUser = users.find((item) => item.id === user?.id) || user;
  const seller = sellers.find((item) => item.userId === user?.id || item.id === user?.sellerId);
  const application = sellerApplications.find((item) => item.userId === user?.id);
  const myOrders = useMemo(
    () => orders.filter((order) => order.customerId === user?.id || order.customerEmail === user?.email),
    [orders, user]
  );
  const sellerProducts = products.filter((product) => product.sellerId === seller?.id);
  const productStats = sellerProducts.map((product) => {
    const unitsSold = orders.reduce((sum, order) => {
      const line = order.items.find((item) => String(item.productId) === String(product.id));
      return sum + (line?.quantity || 0);
    }, 0);
    return {
      ...product,
      unitsSold,
      revenue: unitsSold * product.price,
    };
  });
  const [form, setForm] = useState({
    fullName: storedUser?.fullName || "",
    email: storedUser?.email || "",
    phone: storedUser?.phone || "",
    address: storedUser?.address || "",
    city: storedUser?.city || "",
    zip: storedUser?.zip || "",
  });
  const [saved, setSaved] = useState(false);

  const submit = (event) => {
    event.preventDefault();
    updateUser(user.id, form);
    updateCurrentUser({ fullName: form.fullName, email: form.email });
    setSaved(true);
    setTimeout(() => setSaved(false), 1800);
  };

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <Breadcrumbs items={[{ label: "Mi cuenta" }]} />
      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-black uppercase tracking-[0.18em] text-primary-700">Cuenta AUREX</p>
            <h1 className="mt-2 text-3xl font-black text-slate-950">{storedUser?.fullName}</h1>
            <p className="mt-1 text-slate-600">{storedUser?.email}</p>
          </div>
          <span className="rounded-full bg-slate-100 px-4 py-2 text-sm font-black uppercase text-slate-700">{storedUser?.role}</span>
        </div>
      </section>

      <div className="grid gap-7 lg:grid-cols-[1.2fr,0.8fr]">
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-black text-slate-950">Mi informacion</h2>
          <form onSubmit={submit} className="mt-5 grid gap-4 md:grid-cols-2">
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Nombre completo" value={form.fullName} onChange={(e) => setForm({ ...form, fullName: e.target.value })} required />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" type="email" placeholder="Email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Telefono" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Codigo postal" value={form.zip} onChange={(e) => setForm({ ...form, zip: e.target.value })} />
            <input className="md:col-span-2 rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Direccion" value={form.address} onChange={(e) => setForm({ ...form, address: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Ciudad" value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
            <button className="rounded-full bg-primary-600 px-6 py-3 font-black text-white">Guardar cambios</button>
            {saved && <p className="self-center text-sm font-bold text-emerald-600">Informacion guardada.</p>}
          </form>
        </section>

        <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-black text-slate-950">Vender en AUREX</h2>
          {seller ? (
            <div className="mt-4 space-y-3">
              <p className="text-slate-600">Tu proveedor esta activo como <b>{seller.publicName || seller.companyName}</b>.</p>
              <Link to="/seller/dashboard" className="inline-flex rounded-full bg-primary-600 px-6 py-3 font-black text-white">Ir al panel de vendedor</Link>
            </div>
          ) : application ? (
            <div className="mt-4 space-y-3">
              <p className="text-slate-600">Tu solicitud esta en estado <b>{application.status}</b>.</p>
              <Link to="/seller/onboarding" className="inline-flex rounded-full border-2 border-primary-300 px-6 py-3 font-black text-primary-700">Actualizar informacion</Link>
            </div>
          ) : (
            <div className="mt-4 space-y-3">
              <p className="text-slate-600">Puedes activar tu cuenta como vendedor. AUREX te pedira datos legales, fiscales, contacto y que productos vas a vender.</p>
              <Link to="/seller/onboarding" className="inline-flex rounded-full bg-primary-600 px-6 py-3 font-black text-white">Quiero ser vendedor</Link>
            </div>
          )}
        </section>
      </div>

      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h2 className="text-xl font-black text-slate-950">Mis compras y tickets</h2>
        {myOrders.length === 0 ? (
          <div className="mt-5 rounded-xl border-2 border-dashed border-slate-200 p-8 text-center text-slate-500">
            Todavia no tienes compras registradas.
          </div>
        ) : (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-3">Pedido</th>
                  <th className="py-3">Fecha</th>
                  <th className="py-3">Estado</th>
                  <th className="py-3 text-right">Total</th>
                  <th className="py-3 text-right">Ticket</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {myOrders.map((order) => (
                  <tr key={order.id}>
                    <td className="py-4 font-black text-slate-900">{order.orderNumber}</td>
                    <td className="py-4 text-slate-600">{new Date(order.createdAt).toLocaleString()}</td>
                    <td className="py-4 text-slate-600">{order.status}</td>
                    <td className="py-4 text-right font-bold">{money.format(order.total)}</td>
                    <td className="py-4 text-right">
                      <Link to={`/receipt/${order.id}`} className="font-black text-primary-700 hover:underline">Ver ticket</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {seller && (
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-black text-slate-950">Productos vendidos por {seller.publicName || seller.companyName}</h2>
              <p className="mt-1 text-sm text-slate-600">Stock disponible, ventas acumuladas e ingresos del perfil vendedor.</p>
            </div>
            <Link to="/seller/dashboard" className="rounded-full bg-primary-600 px-5 py-2 text-sm font-black text-white">
              Administrar productos
            </Link>
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-4">
            <div className="rounded-lg bg-slate-50 p-4"><b>{sellerProducts.length}</b><span> productos</span></div>
            <div className="rounded-lg bg-slate-50 p-4"><b>{sellerProducts.reduce((sum, product) => sum + Number(product.stock), 0)}</b><span> en stock</span></div>
            <div className="rounded-lg bg-slate-50 p-4"><b>{productStats.reduce((sum, product) => sum + product.unitsSold, 0)}</b><span> unidades vendidas</span></div>
            <div className="rounded-lg bg-slate-50 p-4"><b>{money.format(productStats.reduce((sum, product) => sum + product.revenue, 0))}</b><span> ventas</span></div>
          </div>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-3">Producto</th>
                  <th className="py-3">SKU</th>
                  <th className="py-3">Seccion</th>
                  <th className="py-3 text-right">Stock</th>
                  <th className="py-3 text-right">Vendido</th>
                  <th className="py-3 text-right">Ventas</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {productStats.map((product) => (
                  <tr key={product.id}>
                    <td className="py-4 font-black text-slate-900">
                      <Link to={`/product/${product.id}`} className="hover:text-primary-700">{product.name}</Link>
                    </td>
                    <td className="py-4 text-slate-600">{product.sku}</td>
                    <td className="py-4 text-slate-600">{categories.find((category) => category.id === product.categoryId)?.name || product.categoryId}</td>
                    <td className={`py-4 text-right font-bold ${product.stock <= product.reorderPoint ? "text-amber-600" : "text-slate-900"}`}>{product.stock}</td>
                    <td className="py-4 text-right">{product.unitsSold}</td>
                    <td className="py-4 text-right font-bold">{money.format(product.revenue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
