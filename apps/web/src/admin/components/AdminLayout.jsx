import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { useLanguage } from "../../context/LanguageContext";
import "../styles/admin.css";

const tabs = [
  { to: "/admin", label: { es: "Estadisticas", en: "Statistics" }, icon: "S", end: true },
  { to: "/admin/customers", label: { es: "Usuarios", en: "Users" }, icon: "U" },
  { to: "/admin/vendors", label: { es: "Proveedores", en: "Providers" }, icon: "P" },
  { to: "/admin/products", label: { es: "Productos", en: "Products" }, icon: "I" },
  { to: "/admin/orders", label: { es: "Ordenes", en: "Orders" }, icon: "O" },
  { to: "/admin/categories", label: { es: "Categorias", en: "Categories" }, icon: "C" },
  { to: "/admin/payments", label: { es: "Pagos", en: "Payments" }, icon: "$" },
  { to: "/admin/inventory", label: { es: "Inventario", en: "Inventory" }, icon: "N" },
  { to: "/admin/analytics", label: { es: "Forecast", en: "Forecast" }, icon: "~" },
];

export function AdminLayout() {
  const { signOut } = useAuth();
  const { language } = useLanguage();
  const isEs = language === "es";

  return (
    <div className="aurex-admin">
      <nav className="aurex-admin-nav">
        <NavLink to="/" className="aurex-admin-logo" aria-label="AUREX home">
          <img src="/aurexlogo-transparent.png" alt="AUREX" className="aurex-admin-logo-img" />
        </NavLink>
        <div className="aurex-admin-actions">
          <NavLink to="/">{isEs ? "Inicio" : "Home"}</NavLink>
          <NavLink to="/contact">{isEs ? "Contacto" : "Contact Us"}</NavLink>
          <NavLink to="/cart">{isEs ? "Carrito" : "Cart"}</NavLink>
          <NavLink to="/admin/settings">{isEs ? "Perfil" : "Profile"}</NavLink>
          <button type="button" className="aurex-admin-logout" onClick={signOut}>
            {isEs ? "Salir" : "Logout"}
          </button>
        </div>
      </nav>

      <main className="aurex-admin-shell">
        <div className="aurex-admin-tabs" aria-label="Admin sections">
          {tabs.map((tab) => (
            <NavLink key={tab.to} to={tab.to} end={tab.end} className="aurex-admin-tab">
              <span>{tab.icon}</span>
              <span>{tab.label[language] || tab.label.en}</span>
            </NavLink>
          ))}
        </div>
        <Outlet />
      </main>
    </div>
  );
}
