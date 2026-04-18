const categoryVisuals = {
  audio: {
    image: "https://images.pexels.com/photos/3394666/pexels-photo-3394666.jpeg?auto=compress&cs=tinysrgb&w=1600",
    accent: "from-cyan-900/85 via-slate-950/55 to-purple-900/35",
    eyebrow: "Audio intelligence",
  },
  workspace: {
    image: "https://images.pexels.com/photos/1957478/pexels-photo-1957478.jpeg?auto=compress&cs=tinysrgb&w=1600",
    accent: "from-slate-950/80 via-blue-950/55 to-cyan-700/25",
    eyebrow: "Smart productivity",
  },
  "smart-home": {
    image: "https://images.pexels.com/photos/3937174/pexels-photo-3937174.jpeg?auto=compress&cs=tinysrgb&w=1600",
    accent: "from-blue-950/80 via-slate-950/55 to-cyan-700/25",
    eyebrow: "Connected living",
  },
  electronics: { image: "https://images.pexels.com/photos/356056/pexels-photo-356056.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-blue-950/55 to-cyan-800/25", eyebrow: "Tech commerce" },
  fashion: { image: "https://images.pexels.com/photos/5709661/pexels-photo-5709661.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-purple-950/45 to-cyan-800/20", eyebrow: "Style catalog" },
  shoes: { image: "https://images.pexels.com/photos/2529148/pexels-photo-2529148.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/45 to-cyan-800/20", eyebrow: "Footwear" },
  beauty: { image: "https://images.pexels.com/photos/3373746/pexels-photo-3373746.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-purple-950/40 to-cyan-800/20", eyebrow: "Beauty demand" },
  home: { image: "https://images.pexels.com/photos/5825577/pexels-photo-5825577.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/50 to-cyan-800/25", eyebrow: "Home essentials" },
  furniture: { image: "https://images.pexels.com/photos/276583/pexels-photo-276583.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/50 to-cyan-800/20", eyebrow: "Furniture planning" },
  kitchen: { image: "https://images.pexels.com/photos/6996085/pexels-photo-6996085.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/50 to-cyan-800/25", eyebrow: "Kitchen supply" },
  grocery: { image: "https://images.pexels.com/photos/264636/pexels-photo-264636.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-emerald-950/35 to-cyan-800/20", eyebrow: "Daily demand" },
  books: { image: "https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/45 to-cyan-800/20", eyebrow: "Learning shelf" },
  toys: { image: "https://images.pexels.com/photos/3662667/pexels-photo-3662667.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-purple-950/45 to-cyan-800/20", eyebrow: "Family demand" },
  sports: { image: "https://images.pexels.com/photos/1552242/pexels-photo-1552242.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/50 to-cyan-800/25", eyebrow: "Sports inventory" },
  automotive: { image: "https://images.pexels.com/photos/3807329/pexels-photo-3807329.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-blue-950/50 to-cyan-800/20", eyebrow: "Auto supply" },
  health: { image: "https://images.pexels.com/photos/3683074/pexels-photo-3683074.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-cyan-950/45 to-blue-800/20", eyebrow: "Wellness catalog" },
  "office-supplies": { image: "https://images.pexels.com/photos/380769/pexels-photo-380769.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/55 to-cyan-800/25", eyebrow: "Office operations" },
  "pet-supplies": { image: "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/45 to-cyan-800/20", eyebrow: "Pet care" },
  garden: { image: "https://images.pexels.com/photos/4505170/pexels-photo-4505170.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-emerald-950/40 to-cyan-800/20", eyebrow: "Outdoor supply" },
  jewelry: { image: "https://images.pexels.com/photos/1457801/pexels-photo-1457801.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-purple-950/35 to-amber-700/20", eyebrow: "Premium catalog" },
  baby: { image: "https://images.pexels.com/photos/3662875/pexels-photo-3662875.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/80 via-blue-950/45 to-cyan-800/20", eyebrow: "Baby essentials" },
  tools: { image: "https://images.pexels.com/photos/162553/keys-workshop-mechanic-tools-162553.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-blue-950/50 to-cyan-800/20", eyebrow: "Tools and hardware" },
  gaming: { image: "https://images.pexels.com/photos/3945654/pexels-photo-3945654.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-purple-950/45 to-cyan-800/20", eyebrow: "Gaming demand" },
  industrial: { image: "https://images.pexels.com/photos/257700/pexels-photo-257700.jpeg?auto=compress&cs=tinysrgb&w=1600", accent: "from-slate-950/85 via-blue-950/55 to-cyan-800/20", eyebrow: "B2B operations" },
  default: {
    image: "https://images.pexels.com/photos/3184465/pexels-photo-3184465.jpeg?auto=compress&cs=tinysrgb&w=1600",
    accent: "from-slate-950/85 via-blue-950/55 to-cyan-800/30",
    eyebrow: "AUREX category",
  },
};

export function getCategoryVisual(category) {
  if (!category) return categoryVisuals.default;
  const key = category.slug || category.name?.toLowerCase().replace(/\s+/g, "-");
  return categoryVisuals[key] || categoryVisuals.default;
}
