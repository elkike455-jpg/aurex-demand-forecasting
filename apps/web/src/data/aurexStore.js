const STORAGE_KEY = "aurex-commerce-state-v1";
export const GENERAL_ADMIN_EMAIL = "elkike455@gmail.com";

export const roles = {
  superadmin: "superadmin",
  admin: "admin",
  staff: "staff",
  seller: "seller",
  customer: "customer",
};

export const orderStatuses = [
  "pending",
  "paid",
  "processing",
  "shipped",
  "delivered",
  "cancelled",
  "refunded",
];

export const paymentStatuses = ["pending", "succeeded", "failed", "refunded"];

const now = () => new Date().toISOString();
const id = (prefix) => `${prefix}_${crypto.randomUUID()}`;
const slugify = (value) =>
  value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");

const commerceCategories = [
  ["cat_electronics", "Electronics", "electronics", "Phones, tablets, computers, and connected devices"],
  ["cat_audio", "Audio", "audio", "Headphones, speakers, microphones, and sound systems"],
  ["cat_smart_home", "Smart Home", "smart-home", "Connected devices, security, lighting, and home automation"],
  ["cat_workspace", "Workspace", "workspace", "Office furniture, lamps, productivity gear, and desk setups"],
  ["cat_fashion", "Fashion", "fashion", "Apparel, everyday wear, and seasonal collections"],
  ["cat_shoes", "Shoes", "shoes", "Sneakers, formal shoes, boots, and sport footwear"],
  ["cat_beauty", "Beauty", "beauty", "Skincare, cosmetics, grooming, and personal care"],
  ["cat_home", "Home", "home", "Home essentials, decor, organization, and living products"],
  ["cat_furniture", "Furniture", "furniture", "Sofas, chairs, tables, beds, and storage"],
  ["cat_kitchen", "Kitchen", "kitchen", "Cookware, appliances, dining, and kitchen tools"],
  ["cat_grocery", "Grocery", "grocery", "Pantry items, beverages, snacks, and daily essentials"],
  ["cat_books", "Books", "books", "Books, learning materials, notebooks, and media"],
  ["cat_toys", "Toys", "toys", "Toys, games, puzzles, and kids entertainment"],
  ["cat_sports", "Sports", "sports", "Fitness, outdoor, training, and sports equipment"],
  ["cat_automotive", "Automotive", "automotive", "Car accessories, tools, parts, and maintenance"],
  ["cat_health", "Health", "health", "Wellness, medical supplies, supplements, and care products"],
  ["cat_office", "Office Supplies", "office-supplies", "Stationery, paper, printers, and business supplies"],
  ["cat_pet", "Pet Supplies", "pet-supplies", "Food, toys, grooming, and pet care"],
  ["cat_garden", "Garden", "garden", "Plants, outdoor furniture, gardening tools, and patio products"],
  ["cat_jewelry", "Jewelry", "jewelry", "Watches, rings, necklaces, and accessories"],
  ["cat_baby", "Baby", "baby", "Baby care, clothing, safety, and nursery essentials"],
  ["cat_tools", "Tools", "tools", "Hand tools, power tools, hardware, and workshop gear"],
  ["cat_gaming", "Gaming", "gaming", "Consoles, accessories, games, streaming, and esports gear"],
  ["cat_industrial", "Industrial", "industrial", "B2B supplies, safety equipment, packaging, and operations"],
].map(([id, name, slug, description]) => ({
  id,
  name,
  slug,
  description,
  createdAt: now(),
  updatedAt: now(),
}));

const defaultSellers = [
  {
    id: "seller_aurex",
    userId: "usr_superadmin",
    companyName: "AUREX Global",
    publicName: "AUREX Global",
    legalName: "AUREX Global Connected Commerce",
    taxId: "AUREX-DEMO-001",
    status: "approved",
    riskStatus: "verified",
    categories: ["cat_electronics", "cat_audio", "cat_smart_home", "cat_workspace"],
    productDescription: "Curated smart commerce catalog managed directly by AUREX.",
    createdAt: now(),
    updatedAt: now(),
  },
  {
    id: "seller_nova",
    userId: "usr_seller",
    companyName: "Nova Retail Labs",
    publicName: "Nova Retail Labs",
    legalName: "Nova Retail Labs LLC",
    taxId: "NOVA-DEMO-2026",
    status: "approved",
    riskStatus: "reviewed",
    categories: ["cat_workspace", "cat_shoes", "cat_electronics"],
    productDescription: "Productivity, lifestyle, and tech accessories for SMEs.",
    createdAt: now(),
    updatedAt: now(),
  },
];

export const seedState = {
  users: [
    {
      id: "usr_superadmin",
      email: GENERAL_ADMIN_EMAIL,
      password: "aurexroot123",
      fullName: "El Kike AUREX General Admin",
      role: roles.superadmin,
      sellerId: "seller_aurex",
      createdAt: now(),
    },
    {
      id: "usr_admin",
      email: "admin@aurex.local",
      password: "admin123",
      fullName: "AUREX Admin",
      role: roles.admin,
      createdAt: now(),
    },
    {
      id: "usr_staff",
      email: "staff@aurex.local",
      password: "staff123",
      fullName: "Inventory Analyst",
      role: roles.staff,
      createdAt: now(),
    },
    {
      id: "usr_seller",
      email: "seller@aurex.local",
      password: "seller123",
      fullName: "Nova Seller",
      role: roles.seller,
      sellerId: "seller_nova",
      createdAt: now(),
    },
    {
      id: "usr_customer",
      email: "customer@aurex.local",
      password: "customer123",
      fullName: "AUREX Customer",
      role: roles.customer,
      createdAt: now(),
    },
  ],
  sellers: defaultSellers,
  sellerApplications: [],
  categories: commerceCategories,
  products: [
    {
      id: 1,
      name: "Wireless Bluetooth Earbuds Pro",
      slug: "wireless-bluetooth-earbuds-pro",
      sku: "AUR-AUD-001",
      categoryId: "cat_audio",
      sellerId: "seller_aurex",
      price: 129.99,
      oldPrice: 169.99,
      stock: 12,
      reorderPoint: 8,
      active: true,
      image: "https://images.pexels.com/photos/373945/pexels-photo-373945.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.7,
      description: "Premium sound quality with active noise cancellation. Up to 24h battery life.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 2,
      name: 'Smartphone 128GB 6.5" Display',
      slug: "smartphone-128gb-display",
      sku: "AUR-MOB-128",
      categoryId: "cat_smart_home",
      sellerId: "seller_aurex",
      price: 699,
      stock: 5,
      reorderPoint: 10,
      active: true,
      image: "https://images.pexels.com/photos/404280/pexels-photo-404280.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.5,
      description: "Fast performance, AMOLED display, triple camera, 5G ready.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 3,
      name: "Running sneakers - midnight edition",
      slug: "running-sneakers-midnight-edition",
      sku: "AUR-LIF-003",
      categoryId: "cat_shoes",
      sellerId: "seller_nova",
      price: 89.99,
      oldPrice: 119.99,
      stock: 22,
      reorderPoint: 9,
      active: true,
      image: "https://images.pexels.com/photos/19090/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.3,
      description: "Lightweight, breathable design with responsive cushioning.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 4,
      name: "LED desk lamp with wireless charger",
      slug: "led-desk-lamp-wireless-charger",
      sku: "AUR-WRK-004",
      categoryId: "cat_workspace",
      sellerId: "seller_nova",
      price: 59.99,
      stock: 3,
      reorderPoint: 12,
      active: true,
      image: "https://images.pexels.com/photos/8091470/pexels-photo-8091470.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.6,
      description: "Adjustable brightness and Qi wireless charging for desk setups.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 5,
      name: "AUREX Smart Inventory Tablet",
      slug: "aurex-smart-inventory-tablet",
      sku: "AUR-ELC-005",
      categoryId: "cat_electronics",
      sellerId: "seller_aurex",
      price: 349,
      oldPrice: 399,
      stock: 18,
      reorderPoint: 7,
      active: true,
      image: "https://images.pexels.com/photos/5082579/pexels-photo-5082579.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.8,
      description: "Tablet para puntos de venta, inventario y administracion de catalogo. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 6,
      name: "AUREX Forecast Barcode Scanner",
      slug: "aurex-forecast-barcode-scanner",
      sku: "AUR-OFF-006",
      categoryId: "cat_office",
      sellerId: "seller_aurex",
      price: 89,
      stock: 32,
      reorderPoint: 10,
      active: true,
      image: "https://images.pexels.com/photos/6153345/pexels-photo-6153345.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.6,
      description: "Scanner ligero para registrar ventas, entradas de almacen y conteos ciclicos. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 7,
      name: "AUREX Mesh WiFi Commerce Router",
      slug: "aurex-mesh-wifi-commerce-router",
      sku: "AUR-SMH-007",
      categoryId: "cat_smart_home",
      sellerId: "seller_aurex",
      price: 159,
      stock: 9,
      reorderPoint: 8,
      active: true,
      image: "https://images.pexels.com/photos/4219862/pexels-photo-4219862.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.4,
      description: "Router mesh para tiendas pequenas con conexion estable para checkout y dispositivos IoT. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 8,
      name: "AUREX Smart Shelf Sensor Kit",
      slug: "aurex-smart-shelf-sensor-kit",
      sku: "AUR-IND-008",
      categoryId: "cat_industrial",
      sellerId: "seller_aurex",
      price: 219,
      stock: 6,
      reorderPoint: 6,
      active: true,
      image: "https://images.pexels.com/photos/4484071/pexels-photo-4484071.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.7,
      description: "Sensores para anaquel que preparan datos de inventario para alertas y forecast. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 9,
      name: "AUREX Thermal Label Printer",
      slug: "aurex-thermal-label-printer",
      sku: "AUR-OFF-009",
      categoryId: "cat_office",
      sellerId: "seller_aurex",
      price: 129,
      stock: 14,
      reorderPoint: 5,
      active: true,
      image: "https://images.pexels.com/photos/4792729/pexels-photo-4792729.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.5,
      description: "Impresora termica para etiquetas de envio, SKU y control de almacen. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: 10,
      name: "AUREX Portable POS Terminal",
      slug: "aurex-portable-pos-terminal",
      sku: "AUR-ELC-010",
      categoryId: "cat_electronics",
      sellerId: "seller_aurex",
      price: 249,
      stock: 11,
      reorderPoint: 6,
      active: true,
      image: "https://images.pexels.com/photos/5799201/pexels-photo-5799201.jpeg?auto=compress&cs=tinysrgb&w=600",
      rating: 4.8,
      description: "Terminal portatil para ventas rapidas, pagos simulados y gestion de pedidos. Vendido por AUREX Global.",
      gallery: [],
      createdAt: now(),
      updatedAt: now(),
    },
  ],
  orders: [
    {
      id: "ord_1001",
      orderNumber: "AUR-1001",
      customerId: "usr_customer",
      customerEmail: "customer@aurex.local",
      status: "paid",
      paymentStatus: "succeeded",
      items: [
        { productId: 1, name: "Wireless Bluetooth Earbuds Pro", sku: "AUR-AUD-001", quantity: 1, price: 129.99 },
        { productId: 4, name: "LED desk lamp with wireless charger", sku: "AUR-WRK-004", quantity: 2, price: 59.99 },
      ],
      subtotal: 249.97,
      shipping: 0,
      total: 249.97,
      currency: "usd",
      notes: "Forecast watch: desk lamps are near reorder point.",
      timeline: [{ status: "paid", at: now(), note: "Payment captured." }],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: "ord_1002",
      orderNumber: "AUR-1002",
      customerId: "usr_customer",
      customerEmail: "customer@aurex.local",
      status: "delivered",
      paymentStatus: "succeeded",
      items: [
        { productId: 5, name: "AUREX Smart Inventory Tablet", sku: "AUR-ELC-005", quantity: 2, price: 349 },
        { productId: 6, name: "AUREX Forecast Barcode Scanner", sku: "AUR-OFF-006", quantity: 3, price: 89 },
      ],
      subtotal: 965,
      shipping: 0,
      total: 965,
      currency: "usd",
      notes: "SME bundle for inventory setup.",
      timeline: [{ status: "delivered", at: now(), note: "Delivered and closed." }],
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: "ord_1003",
      orderNumber: "AUR-1003",
      customerId: "usr_customer",
      customerEmail: "customer@aurex.local",
      status: "paid",
      paymentStatus: "succeeded",
      items: [
        { productId: 8, name: "AUREX Smart Shelf Sensor Kit", sku: "AUR-IND-008", quantity: 1, price: 219 },
        { productId: 10, name: "AUREX Portable POS Terminal", sku: "AUR-ELC-010", quantity: 1, price: 249 },
      ],
      subtotal: 468,
      shipping: 0,
      total: 468,
      currency: "usd",
      notes: "Forecast-ready checkout package.",
      timeline: [{ status: "paid", at: now(), note: "Payment captured." }],
      createdAt: now(),
      updatedAt: now(),
    },
  ],
  payments: [
    {
      id: "pay_1001",
      orderId: "ord_1001",
      provider: "stripe",
      providerTransactionId: "pi_demo_1001",
      amount: 249.97,
      currency: "usd",
      status: "succeeded",
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: "pay_1002",
      orderId: "ord_1002",
      provider: "stripe",
      providerTransactionId: "pi_demo_1002",
      amount: 965,
      currency: "usd",
      status: "succeeded",
      createdAt: now(),
      updatedAt: now(),
    },
    {
      id: "pay_1003",
      orderId: "ord_1003",
      provider: "stripe",
      providerTransactionId: "pi_demo_1003",
      amount: 468,
      currency: "usd",
      status: "succeeded",
      createdAt: now(),
      updatedAt: now(),
    },
  ],
  inventoryMovements: [
    { id: "mov_1", productId: 4, type: "sale", quantity: -2, reason: "Order AUR-1001", createdAt: now() },
    { id: "mov_2", productId: 5, type: "sale", quantity: -2, reason: "Order AUR-1002", createdAt: now() },
    { id: "mov_3", productId: 6, type: "sale", quantity: -3, reason: "Order AUR-1002", createdAt: now() },
    { id: "mov_4", productId: 8, type: "sale", quantity: -1, reason: "Order AUR-1003", createdAt: now() },
    { id: "mov_5", productId: 10, type: "sale", quantity: -1, reason: "Order AUR-1003", createdAt: now() },
  ],
};

export function loadCommerceState() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? migrateCommerceState(JSON.parse(raw)) : seedState;
  } catch {
    return seedState;
  }
}

function mergeById(existing = [], incoming = []) {
  const seen = new Set(existing.map((item) => item.id));
  return [...existing, ...incoming.filter((item) => !seen.has(item.id))];
}

function migrateCommerceState(state) {
  const mergedUsers = mergeById(state.users, seedState.users).map((user) =>
    user.email?.toLowerCase() === GENERAL_ADMIN_EMAIL
      ? { ...user, role: roles.superadmin, sellerId: user.sellerId || "seller_aurex" }
      : user
  );
  const products = mergeById(state.products, seedState.products).map((product) => ({
    ...product,
    sellerId: product.sellerId || "seller_aurex",
  }));

  return {
    ...seedState,
    ...state,
    users: mergedUsers,
    sellers: mergeById(state.sellers, seedState.sellers),
    sellerApplications: state.sellerApplications || [],
    categories: mergeById(state.categories, seedState.categories),
    products,
    orders: mergeById(state.orders, seedState.orders),
    payments: mergeById(state.payments, seedState.payments),
    inventoryMovements: mergeById(state.inventoryMovements, seedState.inventoryMovements),
  };
}

export function persistCommerceState(state) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function makeProduct(input) {
  const numericId = Date.now();
  return {
    id: numericId,
    slug: input.slug || slugify(input.name),
    sku: input.sku || `AUR-${numericId}`,
    gallery: [],
    active: true,
    rating: 0,
    oldPrice: null,
    sellerId: input.sellerId || "seller_aurex",
    createdAt: now(),
    updatedAt: now(),
    ...input,
    price: Number(input.price || 0),
    stock: Number(input.stock || 0),
    reorderPoint: Number(input.reorderPoint || 0),
  };
}

export function makeUser(input) {
  return {
    id: id("usr"),
    email: input.email.trim().toLowerCase(),
    password: input.password,
    fullName: input.fullName,
    role: input.role || roles.customer,
    sellerId: input.sellerId || null,
    createdAt: now(),
  };
}

export function makeSellerApplication({ user, form }) {
  return {
    id: id("seller_app"),
    userId: user.id,
    applicantName: user.fullName,
    applicantEmail: user.email,
    companyName: form.companyName,
    publicName: form.publicName || form.companyName,
    legalName: form.legalName,
    taxId: form.taxId,
    businessType: form.businessType,
    country: form.country,
    state: form.state,
    city: form.city,
    address: form.address,
    phone: form.phone,
    website: form.website,
    categories: form.categories || [],
    productDescription: form.productDescription,
    legalRepresentative: form.legalRepresentative,
    payoutMethod: form.payoutMethod,
    bankAccountLast4: form.bankAccountLast4,
    acceptsTerms: Boolean(form.acceptsTerms),
    status: "pending",
    riskStatus: "needs_review",
    submittedAt: now(),
    updatedAt: now(),
  };
}

export function makeCategory(input) {
  return {
    id: id("cat"),
    name: input.name,
    slug: input.slug || slugify(input.name),
    description: input.description || "",
    createdAt: now(),
    updatedAt: now(),
  };
}

export function makeOrder({ user, items, shipping = 0, paymentStatus = "pending", status = "pending", shippingAddress = null }) {
  const subtotal = items.reduce((sum, item) => sum + item.product.price * item.quantity, 0);
  const orderId = id("ord");
  return {
    id: orderId,
    orderNumber: `AUR-${Math.floor(1000 + Math.random() * 9000)}`,
    customerId: user?.id || "guest",
    customerEmail: user?.email || "guest@aurex.local",
    status,
    paymentStatus,
    items: items.map(({ product, quantity }) => ({
      productId: product.id,
      name: product.name,
      sku: product.sku,
      quantity,
      price: product.price,
    })),
    subtotal,
    shipping,
    shippingAddress,
    total: subtotal + shipping,
    currency: "usd",
    notes: "",
    timeline: [{ status, at: now(), note: "Order created." }],
    createdAt: now(),
    updatedAt: now(),
  };
}

export function makePayment({ order, provider, providerTransactionId, status }) {
  return {
    id: id("pay"),
    orderId: order.id,
    provider,
    providerTransactionId,
    amount: order.total,
    currency: order.currency,
    status,
    createdAt: now(),
    updatedAt: now(),
  };
}

export function getAnalytics(state) {
  const revenue = state.orders
    .filter((order) => ["paid", "processing", "shipped", "delivered"].includes(order.status))
    .reduce((sum, order) => sum + order.total, 0);
  const lowStock = state.products.filter((product) => product.stock <= product.reorderPoint);
  return {
    totalOrders: state.orders.length,
    revenue,
    products: state.products.length,
    customers: state.users.filter((user) => user.role === roles.customer).length,
    sellers: (state.sellers || []).length,
    pendingSellerApplications: (state.sellerApplications || []).filter((application) => application.status === "pending").length,
    lowStockProducts: lowStock.length,
    pendingPayments: state.payments.filter((payment) => payment.status === "pending" || payment.status === "failed").length,
  };
}
