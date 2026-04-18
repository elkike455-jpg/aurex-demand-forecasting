import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useCommerce } from "../context/CommerceContext";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";
import { roles } from "../data/aurexStore";

export function CreateAccountPage() {
  const { registerAccount } = useAuth();
  const { createUser } = useCommerce();
  const navigate = useNavigate();
  const { t } = useLanguage();
  const [form, setForm] = useState({
    fullName: "",
    email: "",
    password: "",
    accountType: roles.customer,
  });
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    if (!/^(?=.*[A-Za-z])(?=.*\d).{8,}$/.test(form.password)) {
      setError("La contrasena debe tener minimo 8 caracteres e incluir letras y numeros.");
      return;
    }
    const user = createUser({
      fullName: form.fullName,
      email: form.email,
      password: form.password,
      role: form.accountType,
    });
    registerAccount(user);
    navigate(form.accountType === roles.seller ? "/seller/onboarding" : "/", { replace: true });
  };

  return (
    <div className="max-w-md mx-auto space-y-6">
      <Breadcrumbs items={[{ label: t("auth.createTitle") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">{t("auth.createTitle")}</h1>
        <p className="text-sm text-slate-600 mb-6">
          {t("auth.createBlurb")}
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setForm((prev) => ({ ...prev, accountType: roles.customer }))}
              className={`rounded-lg border-2 px-4 py-3 text-sm font-bold ${
                form.accountType === roles.customer ? "border-primary-500 bg-primary-50 text-primary-700" : "border-slate-200 text-slate-600"
              }`}
            >
              Consumidor
            </button>
            <button
              type="button"
              onClick={() => setForm((prev) => ({ ...prev, accountType: roles.seller }))}
              className={`rounded-lg border-2 px-4 py-3 text-sm font-bold ${
                form.accountType === roles.seller ? "border-primary-500 bg-primary-50 text-primary-700" : "border-slate-200 text-slate-600"
              }`}
            >
              Vendedor
            </button>
          </div>
          {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">{error}</div>}
          <input
            type="text"
            name="fullName"
            placeholder={t("auth.fullName")}
            value={form.fullName}
            onChange={(e) => setForm((prev) => ({ ...prev, fullName: e.target.value }))}
            required
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <input
            type="email"
            name="email"
            placeholder={t("auth.email")}
            value={form.email}
            onChange={(e) => setForm((prev) => ({ ...prev, email: e.target.value }))}
            required
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <input
            type="password"
            name="password"
            placeholder={t("auth.password")}
            value={form.password}
            onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
            required
            minLength={8}
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <button
            type="submit"
            className="w-full rounded-full bg-gradient-to-r from-primary-500 to-sky-500 py-3 text-base font-bold text-white shadow-md hover:from-primary-400 hover:to-sky-400"
          >
            {t("auth.createAccount")}
          </button>
          {form.accountType === roles.seller && (
            <p className="text-xs font-semibold text-slate-500">
              Al crear una cuenta de vendedor seguiras con el cuestionario legal y comercial para revisar tu alta como proveedor.
            </p>
          )}
        </form>
        <p className="text-sm text-slate-600 mt-5">
          {t("auth.alreadyHave")}{" "}
          <Link to="/sign-in" className="font-semibold text-primary-600 hover:underline">
            {t("auth.signInTitle")}
          </Link>
        </p>
      </div>
    </div>
  );
}

