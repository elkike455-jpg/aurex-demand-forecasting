import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Breadcrumbs } from "../components/Breadcrumbs";

export function SignInPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const redirectTo = location.state?.from?.pathname || "/";

  const handleSubmit = (e) => {
    e.preventDefault();
    const displayName = form.email.split("@")[0] || "AUREX Shopper";
    signIn({ email: form.email, fullName: displayName });
    navigate(redirectTo, { replace: true });
  };

  return (
    <div className="max-w-md mx-auto space-y-6">
      <Breadcrumbs items={[{ label: "Sign in" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Sign in</h1>
        <p className="text-sm text-slate-600 mb-6">
          Access your orders, saved items, and faster checkout.
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={form.email}
            onChange={(e) => setForm((prev) => ({ ...prev, email: e.target.value }))}
            required
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={form.password}
            onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
            required
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <button
            type="submit"
            className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
          >
            Sign in
          </button>
        </form>
        <p className="text-sm text-slate-600 mt-5">
          New to AUREX?{" "}
          <Link to="/create-account" className="font-semibold text-primary-600 hover:underline">
            Create account
          </Link>
        </p>
      </div>
    </div>
  );
}
