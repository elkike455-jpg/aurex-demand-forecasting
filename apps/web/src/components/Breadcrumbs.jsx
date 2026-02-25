import { Link } from "react-router-dom";

/**
 * Items: array of { label, to? }. Last item has no `to` (current page).
 */
export function Breadcrumbs({ items }) {
  return (
    <nav aria-label="Breadcrumb" className="flex flex-wrap items-center gap-2 text-sm text-slate-600 mb-6">
      <Link to="/" className="font-medium text-primary-600 hover:text-primary-700 hover:underline">
        Home
      </Link>
      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-2">
          <span className="text-slate-300">/</span>
          {item.to ? (
            <Link to={item.to} className="font-medium text-primary-600 hover:text-primary-700 hover:underline">
              {item.label}
            </Link>
          ) : (
            <span className="font-semibold text-slate-900" aria-current="page">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
