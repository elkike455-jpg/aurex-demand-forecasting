import { useState, useEffect } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { getReviewsByProductId } from "../../mocks/products";
import { useCart } from "../context/CartContext";
import { ProductCard } from "../components/ProductCard";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";
import { useCommerce } from "../context/CommerceContext";

export function ProductDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { products, sellers } = useCommerce();
  const product = products.find((item) => String(item.id) === String(id));
  const seller = product ? sellers.find((item) => item.id === product.sellerId) : null;
  const { addItem } = useCart();
  const { t, translateProduct, translateReview } = useLanguage();
  const displayProduct = product ? translateProduct(product) : null;
  const [quantity, setQuantity] = useState(1);
  const [added, setAdded] = useState(false);

  useEffect(() => {
    if (displayProduct) document.title = `${displayProduct.name} | AUREX`;
    return () => { document.title = t("productDetail.titleDefault"); };
  }, [displayProduct, t]);

  if (!product) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-slate-800">{t("productDetail.notFound")}</h2>
        <Link to="/products" className="mt-4 inline-block text-primary-600 font-semibold hover:underline">
          {t("productDetail.backToProducts")}
        </Link>
      </div>
    );
  }

  const related = products.filter((item) => String(item.id) !== String(id)).slice(0, 4);
  const reviews = getReviewsByProductId(id).map(translateReview);

  const handleAddToCart = () => {
    addItem(product, quantity);
    setAdded(true);
    setTimeout(() => setAdded(false), 2000);
  };

  return (
    <div className="space-y-12">
      <Breadcrumbs
        items={[
          { label: t("productDetail.products"), to: "/products" },
          { label: displayProduct.name },
        ]}
      />

      <div className="grid gap-8 md:grid-cols-2">
        <div className="relative aspect-square max-h-[480px] rounded-2xl overflow-hidden bg-slate-100 ring-2 ring-slate-100 shadow-lg transition hover:ring-primary-200">
          <img
            src={displayProduct.image}
            alt={displayProduct.name}
            className="w-full h-full object-cover"
          />
        </div>

        <div className="flex flex-col">
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mb-2">
            {displayProduct.name}
          </h1>
          <p className="mb-3 text-sm font-bold uppercase tracking-wide text-primary-700">
            Vendido por {seller?.publicName || seller?.companyName || "AUREX"}
          </p>
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
                {t("products.sale")}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 text-amber-500 mb-6">
            <span className="text-lg">{"â˜…".repeat(Math.round(product.rating))}</span>
            <span className="text-slate-600 font-medium">
              {t("products.ratingOutOf", { rating: product.rating.toFixed(1) })}
            </span>
          </div>
          <p className="text-slate-600 leading-relaxed mb-8">
            {displayProduct.description}
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-700">{t("productDetail.quantity")}</span>
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
              {added ? t("productDetail.added") : t("products.addToCart")}
            </button>
            <button
              onClick={() => {
                addItem(product, quantity);
                navigate("/cart");
              }}
              className="rounded-full border-2 border-slate-300 px-6 py-3 text-base font-semibold text-slate-700 hover:border-primary-500 hover:text-primary-600 active:scale-[0.98] transition"
            >
              {t("productDetail.buyNow")}
            </button>
          </div>
        </div>
      </div>

      {/* Customer reviews */}
      {reviews.length > 0 && (
        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 md:p-8">
          <h2 className="text-xl font-bold text-slate-900 mb-4">{t("productDetail.customerReviews")}</h2>
          <div className="space-y-4">
            {reviews.map((r) => (
              <div key={r.id} className="border-b border-slate-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-slate-800">{r.author}</span>
                  <span className="text-amber-500">{"â˜…".repeat(r.rating)}</span>
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
          <h2 className="text-xl font-bold text-slate-900 mb-4">{t("productDetail.youMayLike")}</h2>
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

