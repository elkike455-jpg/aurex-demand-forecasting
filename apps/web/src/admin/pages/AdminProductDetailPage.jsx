import { Link, useParams } from "react-router-dom";
import { useCommerce } from "../../context/CommerceContext";
import { ForecastHookCard, StatCard } from "../components/AdminCards";

export function AdminProductDetailPage() {
  const { id } = useParams();
  const { products, categories, orders, sellers } = useCommerce();
  const product = products.find((item) => String(item.id) === String(id));

  if (!product) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-8">
        <h1 className="text-2xl font-black">Product not found</h1>
        <Link to="/admin/products" className="mt-4 inline-block font-bold text-cyan-700">Back to products</Link>
      </div>
    );
  }

  const category = categories.find((item) => item.id === product.categoryId);
  const seller = sellers.find((item) => item.id === product.sellerId);
  const unitsSold = orders.reduce((sum, order) => {
    const line = order.items.find((item) => String(item.productId) === String(product.id));
    return sum + (line?.quantity || 0);
  }, 0);

  return (
    <div className="space-y-6">
      <Link to="/admin/products" className="text-sm font-bold text-cyan-700 hover:underline">Back to products</Link>
      <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <img src={product.image} alt={product.name} className="aspect-square w-full rounded-lg object-cover" />
          <h1 className="mt-4 text-2xl font-black text-slate-950">{product.name}</h1>
          <p className="mt-2 text-sm text-slate-500">{product.description}</p>
        </div>
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-4">
            <StatCard label="SKU" value={product.sku} />
            <StatCard label="Category" value={category?.name || "None"} />
            <StatCard label="Seller" value={seller?.publicName || seller?.companyName || "AUREX"} />
            <StatCard label="Stock" value={product.stock} tone={product.stock <= product.reorderPoint ? "amber" : "cyan"} />
            <StatCard label="Units sold" value={unitsSold} tone="navy" />
          </div>
          <div className="grid gap-4 lg:grid-cols-3">
            <ForecastHookCard title="Historical demand chart" />
            <ForecastHookCard title="Forecast curve" />
            <ForecastHookCard title="Reorder recommendation" />
          </div>
        </div>
      </div>
    </div>
  );
}
