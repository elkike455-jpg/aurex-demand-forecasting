import { Link, useParams } from "react-router-dom";
import { ProductCard } from "../components/ProductCard";
import { useCommerce } from "../context/CommerceContext";
import { getCategoryVisual } from "../data/categoryVisuals";

export function CategoryDetailPage() {
  const { slug } = useParams();
  const { categories, products } = useCommerce();
  const category = categories.find((item) => item.slug === slug);
  const visual = getCategoryVisual(category);
  const categoryProducts = products.filter((product) => product.active && product.categoryId === category?.id);

  if (!category) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center">
        <h1 className="text-2xl font-black text-slate-950">Categoría no encontrada</h1>
        <Link to="/categories" className="mt-4 inline-flex rounded-lg bg-primary-600 px-5 py-3 font-bold text-white">
          Ver categorías
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <section className="relative min-h-[360px] overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 shadow-xl">
        <img src={visual.image} alt={category.name} className="absolute inset-0 h-full w-full object-cover" />
        <div className={`absolute inset-0 bg-gradient-to-br ${visual.accent}`} />
        <div className="relative flex min-h-[360px] max-w-3xl flex-col justify-end p-8 text-white md:p-12">
          <Link to="/categories" className="mb-5 w-fit rounded-full bg-white/15 px-4 py-2 text-sm font-bold ring-1 ring-white/25">
            Todas las categorías
          </Link>
          <p className="text-xs font-black uppercase tracking-[0.22em] text-cyan-200">{visual.eyebrow}</p>
          <h1 className="mt-3 text-4xl font-black md:text-5xl">{category.name}</h1>
          <p className="mt-4 text-base leading-7 text-slate-100 md:text-lg">
            {category.description || "Productos seleccionados para compras inteligentes, inventario claro y decisiones rápidas."}
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm font-bold">
            <span className="rounded-lg bg-white/15 px-4 py-2 ring-1 ring-white/20">{categoryProducts.length} productos activos</span>
            <span className="rounded-lg bg-white/15 px-4 py-2 ring-1 ring-white/20">Forecast-ready</span>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-black text-slate-950">Productos de {category.name}</h2>
            <p className="text-sm text-slate-600">Inventario disponible dentro de esta categoría.</p>
          </div>
        </div>
        {categoryProducts.length > 0 ? (
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-4">
            {categoryProducts.map((product) => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-600">
            No hay productos activos en esta categoría todavía.
          </div>
        )}
      </section>
    </div>
  );
}
