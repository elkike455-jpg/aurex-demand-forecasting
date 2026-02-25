import { useMemo, useState } from "react";
import { mockProducts } from "../../mocks/products";
import { ProductCard } from "../components/ProductCard";
import { Breadcrumbs } from "../components/Breadcrumbs";

const SORT_OPTIONS = [
  { value: "featured", label: "Featured" },
  { value: "price-asc", label: "Price: Low to High" },
  { value: "price-desc", label: "Price: High to Low" },
  { value: "rating", label: "Top Rated" },
];

export function ProductListPage() {
  const [sort, setSort] = useState("featured");
  const [onSaleOnly, setOnSaleOnly] = useState(false);

  const filteredAndSorted = useMemo(() => {
    let list = [...mockProducts];
    if (onSaleOnly) list = list.filter((p) => p.oldPrice);
    if (sort === "price-asc") list.sort((a, b) => a.price - b.price);
    else if (sort === "price-desc") list.sort((a, b) => b.price - a.price);
    else if (sort === "rating") list.sort((a, b) => b.rating - a.rating);
    return list;
  }, [sort, onSaleOnly]);

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "All products" }]} />

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h1 className="text-2xl md:text-3xl font-bold text-slate-900">
          All products
        </h1>
        <p className="text-slate-600">
          {filteredAndSorted.length} products
        </p>
      </div>

      {/* Filters & sort */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border-2 border-slate-200 bg-white p-4">
        <span className="text-sm font-semibold text-slate-700">Sort by:</span>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="rounded-lg border-2 border-slate-200 px-4 py-2 text-sm font-medium text-slate-800 bg-white focus:border-primary-500 focus:outline-none cursor-pointer"
        >
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={onSaleOnly}
            onChange={(e) => setOnSaleOnly(e.target.checked)}
            className="rounded border-2 border-slate-300 text-primary-600 focus:ring-primary-500"
          />
          <span className="text-sm font-medium text-slate-700">On sale only</span>
        </label>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
        {filteredAndSorted.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
}
