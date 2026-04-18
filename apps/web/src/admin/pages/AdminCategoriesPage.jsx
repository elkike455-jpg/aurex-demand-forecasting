import { useState } from "react";
import { useCommerce } from "../../context/CommerceContext";
import { AdminTable } from "../components/AdminTable";

export function AdminCategoriesPage() {
  const { categories, products, createCategory, updateCategory, deleteCategory } = useCommerce();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (!name.trim()) return;
    createCategory({ name, description });
    setName("");
    setDescription("");
  };

  return (
    <div className="aurex-panel">
      <h1 className="aurex-admin-title">Categories</h1>
      <form onSubmit={submit} className="aurex-form-card">
        <h2 style={{ color: "#1976d2", fontWeight: 900, marginBottom: 14 }}>Create category</h2>
        <div className="aurex-form-grid">
          <input className="aurex-input" value={name} onChange={(event) => setName(event.target.value)} placeholder="Category name" />
          <input className="aurex-input" value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Description" />
          <button className="aurex-button">Create category</button>
        </div>
      </form>
      <section>
        <AdminTable
          rows={categories}
          getRowKey={(category) => category.id}
          columns={[
            { key: "name", label: "Name" },
            { key: "slug", label: "Slug" },
            { key: "products", label: "Products", render: (category) => products.filter((product) => product.categoryId === category.id).length },
            {
              key: "actions",
              label: "Actions",
              render: (category) => {
                const count = products.filter((product) => product.categoryId === category.id).length;
                return (
                  <div className="flex gap-2">
                    <button onClick={() => updateCategory(category.id, { name: `${category.name}*` })} className="rounded-lg border border-slate-200 px-3 py-1 font-bold">Mark</button>
                    <button disabled={count > 0} onClick={() => deleteCategory(category.id)} className="rounded-lg border border-red-200 px-3 py-1 font-bold text-red-600 disabled:cursor-not-allowed disabled:opacity-40">Delete</button>
                  </div>
                );
              },
            },
          ]}
        />
      </section>
    </div>
  );
}
