const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function toSnakeAddress(address) {
  if (!address) return null;
  return {
    full_name: address.fullName || "",
    email: address.email || "",
    address: address.address || "",
    city: address.city || "",
    zip: address.zip || "",
  };
}

export async function emailReceipt({ order, payment, products, sellers }) {
  const sellerName = (productId) => {
    const product = products.find((item) => String(item.id) === String(productId));
    const seller = sellers.find((item) => item.id === product?.sellerId);
    return seller?.publicName || seller?.companyName || "AUREX Global";
  };

  const payload = {
    order_id: order.id,
    order_number: order.orderNumber,
    customer_email: order.shippingAddress?.email || order.customerEmail,
    customer_name: order.shippingAddress?.fullName || order.customerEmail,
    status: order.status,
    payment_status: payment?.status || order.paymentStatus,
    provider_transaction_id: payment?.providerTransactionId || null,
    currency: order.currency || "usd",
    subtotal: order.subtotal,
    shipping: order.shipping,
    total: order.total,
    created_at: order.createdAt,
    shipping_address: toSnakeAddress(order.shippingAddress),
    items: order.items.map((item) => ({
      product_id: String(item.productId),
      name: item.name,
      sku: item.sku,
      quantity: item.quantity,
      price: item.price,
      seller_name: sellerName(item.productId),
    })),
  };

  const response = await fetch(`${API_BASE_URL}/receipts/email`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "No se pudo enviar el ticket por correo.");
  }

  return response.json();
}
