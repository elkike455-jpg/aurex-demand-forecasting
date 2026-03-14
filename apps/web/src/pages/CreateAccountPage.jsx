import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Breadcrumbs } from "../components/Breadcrumbs";

export function CreateAccountPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    fullName: "",
    email: "",
    password: "",
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    signIn({ email: form.email, fullName: form.fullName });
    navigate("/", { replace: true });
  };

  return (
    <div className="max-w-md mx-auto space-y-6">
      <Breadcrumbs items={[{ label: "Create account" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Create account</h1>
        <p className="text-sm text-slate-600 mb-6">
          Save your profile for quicker orders and order tracking.
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            name="fullName"
            placeholder="Full name"
            value={form.fullName}
            onChange={(e) => setForm((prev) => ({ ...prev, fullName: e.target.value }))}
            required
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
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
            minLength={6}
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <button
            type="submit"
            className="w-full rounded-full bg-gradient-to-r from-primary-500 to-sky-500 py-3 text-base font-bold text-white shadow-md hover:from-primary-400 hover:to-sky-400"
          >
            Create account
          </button>
        </form>
        <p className="text-sm text-slate-600 mt-5">
          Already have an account?{" "}
          <Link to="/sign-in" className="font-semibold text-primary-600 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
