import { Link } from "react-router-dom";
import { useCart } from "../context/CartContext";

export function ProductCard({ product }) {
  const { addItem } = useCart();

  const handleAddToCart = (e) => {
    e.preventDefault();
    e.stopPropagation();
    addItem(product);
  };

  return (
    <Link
      to={`/product/${product.id}`}
      className="bg-white rounded-xl shadow-sm border-2 border-slate-100 hover:shadow-lg hover:border-primary-200 transition-all duration-200 cursor-pointer flex flex-col block card-hover"
    >
      <div className="relative pb-[75%] overflow-hidden rounded-t-xl bg-slate-100">
        <img
          src={product.image}
          alt={product.name}
          className="absolute inset-0 h-full w-full object-cover"
        />
      </div>

      <div className="p-4 flex flex-col gap-2 flex-1">
        <h3 className="text-base font-semibold text-slate-800 leading-snug">{product.name}</h3>

        <div className="flex items-baseline gap-2 mt-1">
          <span className="text-xl font-bold text-slate-900">
            ${product.price.toFixed(2)}
          </span>
          {product.oldPrice && (
            <span className="text-sm text-slate-400 line-through">
              ${product.oldPrice.toFixed(2)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-sm text-amber-500 mt-1">
          <span className="text-base">{"★".repeat(Math.round(product.rating))}</span>
          <span className="text-slate-500 font-medium">
            {product.rating.toFixed(1)} / 5
          </span>
        </div>

        {product.oldPrice && (
          <span className="mt-1 inline-flex w-fit rounded-full bg-primary-50 px-3 py-1 text-xs font-bold text-primary-700">
            SALE
          </span>
        )}

        <button
          type="button"
          onClick={handleAddToCart}
          className="mt-3 w-full rounded-full bg-primary-600 px-4 py-3 text-sm font-semibold text-white hover:bg-primary-700 active:scale-[0.98] transition shadow-sm hover:shadow-md btn-primary-glow"
        >
          Add to cart
        </button>
      </div>
    </Link>
  );
}

