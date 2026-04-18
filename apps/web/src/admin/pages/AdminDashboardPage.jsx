import { useCommerce } from "../../context/CommerceContext";
import { useLanguage } from "../../context/LanguageContext";
import { StatCard } from "../components/AdminCards";

const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export function AdminDashboardPage() {
  const { language } = useLanguage();
  const isEs = language === "es";
  const { analytics, products, categories, orders, payments, sellers } = useCommerce();
  const paidOrders = orders.filter((order) => ["paid", "processing", "shipped", "delivered"].includes(order.status));
  const revenueByMonth = buildRevenueSeries(paidOrders);
  const salesByProduct = buildProductSales(products, paidOrders).slice(0, 8);
  const stockByProduct = products
    .filter((product) => product.sellerId === "seller_aurex")
    .slice(0, 8)
    .map((product) => ({ name: product.name, value: Number(product.stock), danger: product.stock <= product.reorderPoint }));
  const categoryBars = categories
    .map((category) => ({
      name: category.name,
      value: products.filter((product) => product.categoryId === category.id).length,
    }))
    .filter((item) => item.value > 0)
    .slice(0, 10);
  const sellerRows = sellers.map((seller) => {
    const sellerProducts = products.filter((product) => product.sellerId === seller.id);
    const sellerProductIds = new Set(sellerProducts.map((product) => String(product.id)));
    const sellerRevenue = paidOrders.reduce(
      (sum, order) =>
        sum +
        order.items
          .filter((item) => sellerProductIds.has(String(item.productId)))
          .reduce((lineSum, item) => lineSum + item.price * item.quantity, 0),
      0
    );
    return { seller, products: sellerProducts.length, revenue: sellerRevenue };
  });
  const lowStock = products.filter((product) => product.stock <= product.reorderPoint);

  const labels = {
    title: isEs ? "Estadisticas de AUREX" : "AUREX Statistics",
    revenue: isEs ? "Ingresos" : "Revenue",
    products: isEs ? "Productos" : "Products",
    customers: isEs ? "Clientes" : "Customers",
    orders: isEs ? "Ordenes" : "Orders",
    revenueMonth: isEs ? "Ingresos por mes" : "Revenue by month",
    salesProduct: isEs ? "Ventas por producto" : "Sales by product",
    stockProduct: isEs ? "Stock AUREX por producto" : "AUREX stock by product",
    categories: isEs ? "Productos por categoria" : "Products by category",
    providers: isEs ? "Rendimiento por proveedor" : "Provider performance",
    lowStock: isEs ? "Bajo inventario" : "Low stock",
    units: isEs ? "unidades vendidas" : "units sold",
    stock: isEs ? "stock" : "stock",
    seller: isEs ? "Proveedor" : "Provider",
    providerProducts: isEs ? "Productos" : "Products",
    providerRevenue: isEs ? "Ventas" : "Sales",
  };

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">{labels.title}</h1>

      <div className="aurex-admin-widgets">
        <StatCard label={labels.revenue} value={currency.format(analytics.revenue)} color="#0f4c81" icon="$" />
        <StatCard label={labels.products} value={analytics.products} color="#06b6d4" icon="P" />
        <StatCard label={labels.customers} value={analytics.customers} color="#f59e0b" icon="U" />
        <StatCard label={labels.orders} value={analytics.totalOrders} color="#1976d2" icon="O" />
      </div>

      <section className="aurex-chart-card">
        <h3 className="aurex-chart-title" style={{ color: "#0f4c81" }}>{labels.revenueMonth}</h3>
        <svg className="aurex-area" viewBox="0 0 900 280" role="img" aria-label={labels.revenueMonth}>
          <defs>
            <linearGradient id="aurexRevenueArea" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#0284c7" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.18" />
            </linearGradient>
          </defs>
          {[45, 90, 135, 180, 225].map((y) => <line key={y} x1="48" x2="870" y1={y} y2={y} stroke="#e5e7eb" strokeDasharray="4 4" />)}
          <path d={areaPath(revenueByMonth)} fill="url(#aurexRevenueArea)" stroke="#0284c7" strokeWidth="3" />
          {revenueByMonth.map((point, index) => (
            <g key={point.month}>
              <circle cx={80 + index * 155} cy={valueY(point.value, revenueByMonth)} r="5" fill="#0f4c81" />
              <text x={80 + index * 155} y="262" textAnchor="middle" fontSize="12" fill="#64748b">{point.month}</text>
              <text x={80 + index * 155} y={valueY(point.value, revenueByMonth) - 12} textAnchor="middle" fontSize="12" fill="#0f172a">{currency.format(point.value)}</text>
            </g>
          ))}
        </svg>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <BarChart title={labels.salesProduct} rows={salesByProduct} color="#7c3aed" suffix={` ${labels.units}`} />
        <BarChart title={labels.stockProduct} rows={stockByProduct} color="#06b6d4" dangerColor="#f59e0b" suffix={` ${labels.stock}`} />
        <BarChart title={labels.categories} rows={categoryBars} color="#1976d2" />
        <section className="aurex-chart-card">
          <h3 className="aurex-chart-title" style={{ color: "#0f4c81" }}>{labels.providers}</h3>
          <div className="overflow-x-auto">
            <table className="aurex-table">
              <thead>
                <tr>
                  <th>{labels.seller}</th>
                  <th>{labels.providerProducts}</th>
                  <th>{labels.providerRevenue}</th>
                </tr>
              </thead>
              <tbody>
                {sellerRows.map((row) => (
                  <tr key={row.seller.id}>
                    <td>{row.seller.publicName || row.seller.companyName}</td>
                    <td>{row.products}</td>
                    <td>{currency.format(row.revenue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="aurex-chart-card">
        <h3 className="aurex-chart-title" style={{ color: "#f59e0b" }}>{labels.lowStock}</h3>
        <div className="grid gap-3 md:grid-cols-3">
          {lowStock.map((product) => (
            <div key={product.id} className="rounded-lg border border-amber-200 bg-amber-50 p-4">
              <p className="font-black text-slate-900">{product.name}</p>
              <p className="mt-1 text-sm text-slate-600">SKU {product.sku}</p>
              <p className="mt-2 text-sm font-bold text-amber-700">{product.stock} / reorder {product.reorderPoint}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function BarChart({ title, rows, color, dangerColor, suffix = "" }) {
  const max = Math.max(1, ...rows.map((row) => row.value));
  return (
    <section className="aurex-chart-card">
      <h3 className="aurex-chart-title" style={{ color }}>{title}</h3>
      <div className="space-y-3">
        {rows.map((row) => (
          <div key={row.name}>
            <div className="mb-1 flex justify-between gap-3 text-xs font-bold text-slate-600">
              <span className="truncate">{row.name}</span>
              <span>{row.value}{suffix}</span>
            </div>
            <div className="h-4 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.max(6, (row.value / max) * 100)}%`,
                  background: row.danger ? dangerColor || color : color,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function buildProductSales(products, orders) {
  return products.map((product) => {
    const value = orders.reduce((sum, order) => {
      const line = order.items.find((item) => String(item.productId) === String(product.id));
      return sum + (line?.quantity || 0);
    }, 0);
    return { name: product.name, value };
  }).filter((item) => item.value > 0).sort((a, b) => b.value - a.value);
}

function buildRevenueSeries(orders) {
  const months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"];
  const totals = months.map((month) => ({ month, value: 0 }));
  orders.forEach((order, index) => {
    const bucket = Math.min(totals.length - 1, index);
    totals[bucket].value += order.total;
  });
  return totals;
}

function valueY(value, points) {
  const max = Math.max(1, ...points.map((point) => point.value));
  return 225 - (value / max) * 175;
}

function areaPath(points) {
  const coords = points.map((point, index) => [80 + index * 155, valueY(point.value, points)]);
  const line = coords.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
  const last = coords[coords.length - 1];
  const first = coords[0];
  return `${line} L ${last[0]} 235 L ${first[0]} 235 Z`;
}
