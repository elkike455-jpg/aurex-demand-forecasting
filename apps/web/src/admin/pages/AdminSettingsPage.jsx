import { Link } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { useCommerce } from "../../context/CommerceContext";

export function AdminSettingsPage() {
  const { user } = useAuth();
  const { analytics, sellers, sellerApplications } = useCommerce();
  const isRoot = user?.role === "superadmin";

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-black text-slate-950">Settings</h1>

      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-black">{isRoot ? "AUREX General Admin" : "Admin profile"}</h2>
            <p className="mt-2 text-sm text-slate-600">
              {isRoot
                ? "Perfil maximo de AUREX: puede ver proveedores, usuarios, inventario, pagos, forecast hooks y tambien vender con AUREX Global."
                : "Perfil operativo con acceso administrativo segun su rol."}
            </p>
          </div>
          <span className={`rounded-full px-4 py-2 text-sm font-black ${isRoot ? "bg-primary-50 text-primary-700" : "bg-slate-100 text-slate-700"}`}>
            {user?.role}
          </span>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <div className="rounded-lg bg-slate-50 p-4"><b>{analytics.totalOrders}</b><span> Orders</span></div>
          <div className="rounded-lg bg-slate-50 p-4"><b>{sellers.length}</b><span> Providers</span></div>
          <div className="rounded-lg bg-slate-50 p-4"><b>{sellerApplications.filter((item) => item.status === "pending").length}</b><span> Pending providers</span></div>
          <div className="rounded-lg bg-slate-50 p-4"><b>{analytics.lowStockProducts}</b><span> Low stock</span></div>
        </div>
        {isRoot && (
          <div className="mt-5 flex flex-wrap gap-3">
            <Link to="/admin/vendors" className="rounded-full bg-primary-600 px-5 py-2 text-sm font-bold text-white">Ver proveedores</Link>
            <Link to="/seller/dashboard" className="rounded-full border-2 border-primary-300 px-5 py-2 text-sm font-bold text-primary-700">Vender como AUREX</Link>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-black">Security checklist</h2>
        <ul className="mt-4 space-y-2 text-sm text-slate-600">
          <li>Login demo protegido con codigo de 6 digitos y bloqueo temporal tras 5 intentos fallidos.</li>
          <li>Roles separados: superadmin, admin, staff, seller y customer.</li>
          <li>Rutas admin y seller protegidas por rol en el router.</li>
          <li>Registro de vendedor con datos legales, fiscales, cuenta de pago y revision manual.</li>
          <li>En produccion: mover sesiones, OTP, hash de password, RBAC, rate limit y auditoria al backend.</li>
        </ul>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-black">Environment checklist</h2>
        <ul className="mt-4 space-y-2 text-sm text-slate-600">
          <li>STRIPE_SECRET_KEY configured on backend only.</li>
          <li>STRIPE_WEBHOOK_SECRET configured for webhook verification.</li>
          <li>VITE_API_BASE_URL points to the AUREX API service.</li>
          <li>Admin credentials seeded server-side before production.</li>
        </ul>
      </section>
    </div>
  );
}
