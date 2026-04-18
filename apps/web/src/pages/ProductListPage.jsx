import { useMemo, useState } from "react";
import { ProductCard } from "../components/ProductCard";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";
import { useCommerce } from "../context/CommerceContext";

const SORT_OPTIONS = [
  { value: "featured", labelKey: "productList.sortOptions.featured" },
  { value: "price-asc", labelKey: "productList.sortOptions.price-asc" },
  { value: "price-desc", labelKey: "productList.sortOptions.price-desc" },
  { value: "rating", labelKey: "productList.sortOptions.rating" },
];

export function ProductListPage() {
  const [sort, setSort] = useState("featured");
  const [onSaleOnly, setOnSaleOnly] = useState(false);
  const [tag, setTag] = useState("all");
  const { t, translateProduct } = useLanguage();
  const { products } = useCommerce();
  const tags = t("productList.tags");

  const filteredAndSorted = useMemo(() => {
    let list = products.filter((product) => product.active);
    if (onSaleOnly) list = list.filter((p) => p.oldPrice);
    if (tag !== "all") {
      const lower = tags[Number(tag)].toLowerCase();
      list = list.filter((p) => translateProduct(p).name.toLowerCase().includes(lower));
    }
    if (sort === "price-asc") list.sort((a, b) => a.price - b.price);
    else if (sort === "price-desc") list.sort((a, b) => b.price - a.price);
    else if (sort === "rating") list.sort((a, b) => b.rating - a.rating);
    return list;
  }, [sort, onSaleOnly, tag, tags, translateProduct, products]);

  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: t("productList.title") }]} />

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900">{t("productList.title")}</h1>
          <p className="text-slate-600 text-sm">{t("productList.description")}</p>
        </div>
        <p className="text-slate-600">
          {t(filteredAndSorted.length === 1 ? "productList.productCount_one" : "productList.productCount_other", {
            count: filteredAndSorted.length,
          })}
        </p>
      </div>

      {/* Filters & sort */}
      <div className="rounded-xl border-2 border-slate-200 bg-white p-4 space-y-4">
        <div className="flex flex-wrap gap-2">
          {tags.map((tagLabel, index) => (
            <button
              key={tagLabel}
              onClick={() => setTag(index === 0 ? "all" : String(index))}
              className={`rounded-full px-4 py-2 text-sm font-semibold border ${
                tag === (index === 0 ? "all" : String(index))
                  ? "border-primary-400 bg-primary-50 text-primary-700"
                  : "border-slate-200 text-slate-700 hover:border-primary-200 hover:text-primary-700"
              }`}
            >
              {tagLabel}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-sm font-semibold text-slate-700">{t("productList.sortBy")}</span>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="rounded-lg border-2 border-slate-200 px-4 py-2 text-sm font-medium text-slate-800 bg-white focus:border-primary-500 focus:outline-none cursor-pointer"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{t(opt.labelKey)}</option>
            ))}
          </select>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={onSaleOnly}
              onChange={(e) => setOnSaleOnly(e.target.checked)}
              className="rounded border-2 border-slate-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm font-medium text-slate-700">{t("productList.onSaleOnly")}</span>
          </label>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
        {filteredAndSorted.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
}

