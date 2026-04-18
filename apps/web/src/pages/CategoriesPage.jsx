import { Link } from "react-router-dom";
import { useCommerce } from "../context/CommerceContext";
import { getCategoryVisual } from "../data/categoryVisuals";

export function CategoriesPage() {
  const { categories, products } = useCommerce();

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm font-bold uppercase tracking-[0.18em] text-primary-600">AUREX Catalog</p>
        <h1 className="mt-2 text-3xl font-black text-slate-950">Categorías</h1>
        <p className="mt-2 max-w-2xl text-slate-600">
          Explora los productos por familia. Cada categoría concentra inventario, productos activos y señales futuras para análisis de demanda.
        </p>
      </section>

      <section className="grid gap-5 md:grid-cols-3">
        {categories.map((category) => {
          const visual = getCategoryVisual(category);
          const activeCount = products.filter((product) => product.categoryId === category.id && product.active).length;
          return (
            <Link
              key={category.id}
              to={`/category/${category.slug}`}
              className="group relative min-h-[260px] overflow-hidden rounded-xl border border-slate-200 bg-slate-900 shadow-sm transition hover:-translate-y-1 hover:shadow-xl"
            >
              <img
                src={visual.image}
                alt={category.name}
                className="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-105"
              />
              <div className={`absolute inset-0 bg-gradient-to-br ${visual.accent}`} />
              <div className="relative flex h-full min-h-[260px] flex-col justify-end p-6 text-white">
                <p className="text-xs font-black uppercase tracking-[0.2em] text-cyan-200">{visual.eyebrow}</p>
                <h2 className="mt-2 text-2xl font-black">{category.name}</h2>
                <p className="mt-2 text-sm text-slate-100">{category.description || "Productos seleccionados por AUREX."}</p>
                <div className="mt-5 inline-flex w-fit rounded-lg bg-white/15 px-4 py-2 text-sm font-bold ring-1 ring-white/25">
                  {activeCount} productos
                </div>
              </div>
            </Link>
          );
        })}
      </section>
    </div>
  );
}
