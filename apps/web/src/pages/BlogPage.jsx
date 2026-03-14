import { Breadcrumbs } from "../components/Breadcrumbs";

const posts = [
  { id: 1, title: "Designing a future-ready workspace", tag: "Workspace", readTime: "6 min" },
  { id: 2, title: "How to layer lighting for any room", tag: "Lighting", readTime: "5 min" },
  { id: 3, title: "Smart home without the headache", tag: "Smart home", readTime: "7 min" },
];

export function BlogPage() {
  return (
    <div className="space-y-8">
      <Breadcrumbs items={[{ label: "Blog" }]} />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {posts.map((post) => (
          <article key={post.id} className="rounded-2xl border-2 border-slate-200 bg-white p-5 shadow-sm flex flex-col gap-3">
            <span className="text-xs font-semibold text-primary-600 uppercase tracking-[0.14em]">{post.tag}</span>
            <h2 className="text-lg font-bold text-slate-900">{post.title}</h2>
            <p className="text-sm text-slate-500">{post.readTime} read • Editorial</p>
            <button className="self-start rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">Read</button>
          </article>
        ))}
      </div>
    </div>
  );
}
