import { useState, useEffect } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { getProductById, getRelatedProducts, getReviewsByProductId } from "../../mocks/products";
import { useCart } from "../context/CartContext";
import { ProductCard } from "../components/ProductCard";
import { Breadcrumbs } from "../components/Breadcrumbs";

export function ProductDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const product = getProductById(id);
  const { addItem } = useCart();
  const [quantity, setQuantity] = useState(1);
  const [added, setAdded] = useState(false);

  useEffect(() => {
    if (product) document.title = `${product.name} | AUREX`;
    return () => { document.title = "AUREX | Futuristic Marketplace"; };
  }, [product]);

  if (!product) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-slate-800">Product not found</h2>
        <Link to="/products" className="mt-4 inline-block text-primary-600 font-semibold hover:underline">
          Back to all products
        </Link>
      </div>
    );
  }

  const related = getRelatedProducts(id);
  const reviews = getReviewsByProductId(id);

  const handleAddToCart = () => {
    addItem(product, quantity);
    setAdded(true);
    setTimeout(() => setAdded(false), 2000);
  };

  return (
    <div className="space-y-12">
      <Breadcrumbs
        items={[
          { label: "Products", to: "/products" },
          { label: product.name },
        ]}
      />

      <div className="grid gap-8 md:grid-cols-2">
        <div className="relative aspect-square max-h-[480px] rounded-2xl overflow-hidden bg-slate-100 ring-2 ring-slate-100 shadow-lg transition hover:ring-primary-200">
          <img
            src={product.image}
            alt={product.name}
            className="w-full h-full object-cover"
          />
        </div>

        <div className="flex flex-col">
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mb-2">
            {product.name}
          </h1>
          <div className="flex items-center gap-3 mb-4">
            <span className="text-2xl font-bold text-slate-900">
              ${product.price.toFixed(2)}
            </span>
            {product.oldPrice && (
              <span className="text-lg text-slate-400 line-through">
                ${product.oldPrice.toFixed(2)}
              </span>
            )}
            {product.oldPrice && (
              <span className="rounded-full bg-primary-100 px-3 py-1 text-sm font-bold text-primary-700">
                SALE
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 text-amber-500 mb-6">
            <span className="text-lg">{"★".repeat(Math.round(product.rating))}</span>
            <span className="text-slate-600 font-medium">
              {product.rating.toFixed(1)} / 5
            </span>
          </div>
          <p className="text-slate-600 leading-relaxed mb-8">
            {product.description}
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-700">Quantity:</span>
              <input
                type="number"
                min={1}
                max={99}
                value={quantity}
                onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value, 10) || 1))}
                className="w-20 rounded-lg border-2 border-slate-200 px-3 py-2 text-center font-semibold focus:border-primary-500 focus:outline-none"
              />
            </label>
            <button
              onClick={handleAddToCart}
              className="rounded-full bg-primary-600 px-8 py-3 text-base font-bold text-white hover:bg-primary-700 active:scale-[0.98] transition shadow-md hover:shadow-lg"
            >
              {added ? "Added! ✓" : "Add to cart"}
            </button>
            <button
              onClick={() => {
                addItem(product, quantity);
                navigate("/cart");
              }}
              className="rounded-full border-2 border-slate-300 px-6 py-3 text-base font-semibold text-slate-700 hover:border-primary-500 hover:text-primary-600 active:scale-[0.98] transition"
            >
              Buy now
            </button>
          </div>
        </div>
      </div>

      {/* Customer reviews */}
      {reviews.length > 0 && (
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Customer reviews</h2>
          <div className="space-y-4">
            {reviews.map((r) => (
              <div key={r.id} className="border-b border-slate-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-slate-800">{r.author}</span>
                  <span className="text-amber-500">{"★".repeat(r.rating)}</span>
                  <span className="text-sm text-slate-400">{r.date}</span>
                </div>
                <p className="text-slate-600 text-sm">{r.text}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Related products */}
      {related.length > 0 && (
        <section>
          <h2 className="text-xl font-bold text-slate-900 mb-4">You may also like</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
            {related.map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
