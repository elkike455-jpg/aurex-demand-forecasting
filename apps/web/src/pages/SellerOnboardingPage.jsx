import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Breadcrumbs } from "../components/Breadcrumbs";
import { useAuth } from "../context/AuthContext";
import { useCommerce } from "../context/CommerceContext";

const emptyForm = {
  companyName: "",
  publicName: "",
  legalName: "",
  taxId: "",
  businessType: "company",
  country: "",
  state: "",
  city: "",
  address: "",
  phone: "",
  website: "",
  categories: [],
  productDescription: "",
  legalRepresentative: "",
  payoutMethod: "bank_transfer",
  bankAccountLast4: "",
  acceptsTerms: false,
};

export function SellerOnboardingPage() {
  const { user } = useAuth();
  const { categories, submitSellerApplication } = useCommerce();
  const navigate = useNavigate();
  const [form, setForm] = useState(emptyForm);
  const [submitted, setSubmitted] = useState(false);

  if (!user) {
    return (
      <div className="max-w-2xl mx-auto rounded-2xl border-2 border-slate-200 bg-white p-8 text-center">
        <h1 className="text-2xl font-black text-slate-900">Primero inicia sesion</h1>
        <p className="mt-2 text-slate-600">Necesitas una cuenta para registrarte como vendedor.</p>
        <Link to="/sign-in" className="mt-5 inline-flex rounded-full bg-primary-600 px-6 py-3 font-bold text-white">
          Entrar
        </Link>
      </div>
    );
  }

  const toggleCategory = (categoryId) => {
    setForm((current) => ({
      ...current,
      categories: current.categories.includes(categoryId)
        ? current.categories.filter((id) => id !== categoryId)
        : [...current.categories, categoryId],
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!form.acceptsTerms || form.categories.length === 0) return;
    submitSellerApplication({ user, form });
    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="max-w-3xl mx-auto rounded-2xl border-2 border-cyan-100 bg-white p-8 shadow-sm">
        <Breadcrumbs items={[{ label: "Alta de vendedor" }]} />
        <h1 className="text-3xl font-black text-slate-950">Solicitud enviada</h1>
        <p className="mt-3 text-slate-600">
          AUREX recibio tus datos legales y comerciales. El admin general puede revisarlos en proveedores y aprobar tu cuenta.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/seller/dashboard" className="rounded-full bg-primary-600 px-6 py-3 font-bold text-white">
            Ir al panel de vendedor
          </Link>
          <button type="button" onClick={() => navigate("/")} className="rounded-full border-2 border-slate-300 px-6 py-3 font-bold text-slate-700">
            Volver a la tienda
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <Breadcrumbs items={[{ label: "Alta de vendedor" }]} />
      <section className="rounded-2xl border-2 border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-3xl font-black text-slate-950">Registro de proveedor AUREX</h1>
        <p className="mt-2 text-slate-600">
          Completa tu informacion legal, comercial y operativa. Esto ayuda a proteger compradores, vendedores y el marketplace.
        </p>
        <form onSubmit={handleSubmit} className="mt-7 space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Nombre de empresa" value={form.companyName} onChange={(e) => setForm({ ...form, companyName: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Nombre publico del vendedor" value={form.publicName} onChange={(e) => setForm({ ...form, publicName: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Razon social / nombre legal" value={form.legalName} onChange={(e) => setForm({ ...form, legalName: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="RFC / Tax ID / identificacion fiscal" value={form.taxId} onChange={(e) => setForm({ ...form, taxId: e.target.value })} />
            <select className="rounded-lg border-2 border-slate-200 px-4 py-3" value={form.businessType} onChange={(e) => setForm({ ...form, businessType: e.target.value })}>
              <option value="company">Empresa registrada</option>
              <option value="individual">Persona fisica / independiente</option>
              <option value="manufacturer">Fabricante</option>
              <option value="distributor">Distribuidor</option>
            </select>
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Representante legal" value={form.legalRepresentative} onChange={(e) => setForm({ ...form, legalRepresentative: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Pais" value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Estado" value={form.state} onChange={(e) => setForm({ ...form, state: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Ciudad" value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Direccion fiscal / operativa" value={form.address} onChange={(e) => setForm({ ...form, address: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required placeholder="Telefono" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" placeholder="Sitio web o red social" value={form.website} onChange={(e) => setForm({ ...form, website: e.target.value })} />
            <select className="rounded-lg border-2 border-slate-200 px-4 py-3" value={form.payoutMethod} onChange={(e) => setForm({ ...form, payoutMethod: e.target.value })}>
              <option value="bank_transfer">Transferencia bancaria</option>
              <option value="stripe_connect">Stripe Connect</option>
              <option value="manual_review">Revision manual</option>
            </select>
            <input className="rounded-lg border-2 border-slate-200 px-4 py-3" required maxLength={4} placeholder="Ultimos 4 digitos de cuenta" value={form.bankAccountLast4} onChange={(e) => setForm({ ...form, bankAccountLast4: e.target.value.replace(/\D/g, "") })} />
          </div>

          <div>
            <h2 className="text-lg font-black text-slate-900">Que vas a vender</h2>
            <p className="text-sm text-slate-500">Elige las categorias principales. Puedes elegir aunque todavia no tengas productos publicados.</p>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {categories.map((category) => (
                <button
                  key={category.id}
                  type="button"
                  onClick={() => toggleCategory(category.id)}
                  className={`rounded-lg border-2 px-3 py-2 text-left text-sm font-bold ${
                    form.categories.includes(category.id) ? "border-primary-500 bg-primary-50 text-primary-700" : "border-slate-200 text-slate-600"
                  }`}
                >
                  {category.name}
                </button>
              ))}
            </div>
          </div>

          <textarea
            required
            rows={5}
            className="w-full rounded-lg border-2 border-slate-200 px-4 py-3"
            placeholder="Describe tus productos, origen, marcas, capacidad de inventario, tiempos de envio y controles de calidad."
            value={form.productDescription}
            onChange={(e) => setForm({ ...form, productDescription: e.target.value })}
          />

          <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
            <input type="checkbox" checked={form.acceptsTerms} onChange={(e) => setForm({ ...form, acceptsTerms: e.target.checked })} className="mt-1" />
            Confirmo que la informacion es real, que puedo vender legalmente estos productos y acepto revision antifraude, cumplimiento fiscal, terminos de marketplace y politicas de AUREX.
          </label>

          <button type="submit" className="rounded-full bg-primary-600 px-8 py-3 font-black text-white hover:bg-primary-700">
            Enviar solicitud de vendedor
          </button>
        </form>
      </section>
    </div>
  );
}
