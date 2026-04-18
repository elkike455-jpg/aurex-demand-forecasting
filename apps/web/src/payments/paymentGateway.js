const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const STRIPE_ENABLED = import.meta.env.VITE_STRIPE_ENABLED === "true";

export class PaymentGatewayError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = "PaymentGatewayError";
    this.cause = cause;
  }
}

export async function createCheckoutPayment({ order }) {
  if (STRIPE_ENABLED && API_BASE_URL) {
    return createStripeCheckoutSession(order);
  }

  return {
    provider: "stripe",
    providerTransactionId: `pi_local_${order.id}`,
    status: "succeeded",
    redirectUrl: null,
  };
}

async function createStripeCheckoutSession(order) {
  try {
    const response = await fetch(`${API_BASE_URL}/payments/stripe/checkout-session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        order_id: order.id,
        customer_email: order.customerEmail,
        currency: order.currency,
        items: order.items.map((item) => ({
          product_id: String(item.productId),
          name: item.name,
          sku: item.sku || String(item.productId),
          quantity: item.quantity,
          price: item.price,
        })),
      }),
    });

    if (!response.ok) {
      throw new Error(`Stripe checkout failed with ${response.status}`);
    }

    const payload = await response.json();
    if (!payload.url) {
      throw new Error("Stripe checkout response did not include a redirect URL");
    }

    return {
      provider: "stripe",
      providerTransactionId: payload.sessionId,
      status: "pending",
      redirectUrl: payload.url,
    };
  } catch (error) {
    throw new PaymentGatewayError("Unable to start Stripe checkout", error);
  }
}
