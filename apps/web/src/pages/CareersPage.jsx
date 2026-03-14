import { Breadcrumbs } from "../components/Breadcrumbs";

const roles = [
  { title: "Product Designer", type: "Remote", level: "Mid-Senior" },
  { title: "Frontend Engineer", type: "Remote", level: "Senior" },
  { title: "Customer Experience Lead", type: "US", level: "Mid" },
];

export function CareersPage() {
  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Careers" }]} />
      <div className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <h1 className="text-2xl font-bold text-slate-900">Join the team</h1>
        <p className="text-sm text-slate-600">We build a seamless shopping experience for modern homes. Competitive benefits, remote-friendly.</p>
        <div className="space-y-3">
          {roles.map((role) => (
            <div key={role.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">{role.title}</h3>
                <p className="text-sm text-slate-600">{role.type} • {role.level}</p>
              </div>
              <button className="rounded-full bg-primary-600 px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700">Apply</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
