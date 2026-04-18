import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useLanguage } from "../context/LanguageContext";

export function SignInPage() {
  const { signIn, verifySignInCode } = useAuth();
  const { t } = useLanguage();
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });
  const [verification, setVerification] = useState(null);
  const [code, setCode] = useState("");
  const [error, setError] = useState("");

  const redirectTo = location.state?.from?.pathname || "/";

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    try {
      const displayName = form.email.split("@")[0] || t("auth.shopperFallback");
      const result = signIn({ email: form.email, fullName: displayName, password: form.password });
      if (result?.requiresCode) {
        setVerification(result);
        return;
      }
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err.message || "No se pudo iniciar sesion.");
    }
  };

  const handleVerify = (e) => {
    e.preventDefault();
    setError("");
    try {
      verifySignInCode(code);
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err.message || "Codigo invalido.");
    }
  };

  return (
    <div className="max-w-md mx-auto space-y-6">
      <Breadcrumbs items={[{ label: t("auth.signInTitle") }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">{t("auth.signInTitle")}</h1>
        <p className="text-sm text-slate-600 mb-6">
          {t("auth.signInBlurb")}
        </p>
        {error && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
            {error}
          </div>
        )}
        {!verification ? (
        <form onSubmit={handleSubmit} className="space-y-4">
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
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
          />
          <button
            type="submit"
            className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
          >
            {t("auth.signInTitle")}
          </button>
        </form>
        ) : (
          <form onSubmit={handleVerify} className="space-y-4">
            <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm text-slate-700">
              <p className="font-bold text-slate-900">Verificacion de seguridad</p>
              <p>Codigo demo: <span className="font-black text-primary-700">{verification.demoCode}</span></p>
              <p>En produccion este codigo se manda por correo, SMS o app autenticadora.</p>
            </div>
            <input
              type="text"
              inputMode="numeric"
              maxLength={6}
              placeholder="Codigo de 6 digitos"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              required
              className="w-full rounded-lg border-2 border-slate-200 px-4 py-3 text-base focus:border-primary-500 focus:outline-none"
            />
            <button
              type="submit"
              className="w-full rounded-full bg-primary-600 py-3 text-base font-bold text-white hover:bg-primary-700 transition-colors"
            >
              Verificar y entrar
            </button>
          </form>
        )}
        <p className="text-sm text-slate-600 mt-5">
          {t("auth.newTo")}{" "}
          <Link to="/create-account" className="font-semibold text-primary-600 hover:underline">
            {t("auth.createAccount")}
          </Link>
        </p>
      </div>
    </div>
  );
}

