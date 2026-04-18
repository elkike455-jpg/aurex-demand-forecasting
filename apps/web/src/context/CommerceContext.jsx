import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  getAnalytics,
  loadCommerceState,
  makeCategory,
  makeSellerApplication,
  makeOrder,
  makePayment,
  makeProduct,
  makeUser,
  persistCommerceState,
  roles,
} from "../data/aurexStore";

const CommerceContext = createContext(null);

export function CommerceProvider({ children }) {
  const [state, setState] = useState(loadCommerceState);

  useEffect(() => {
    persistCommerceState(state);
  }, [state]);

  const updateProduct = (productId, changes) => {
    setState((current) => ({
      ...current,
      products: current.products.map((product) =>
        String(product.id) === String(productId)
          ? {
              ...product,
              ...changes,
              price: Number(changes.price ?? product.price),
              stock: Number(changes.stock ?? product.stock),
              reorderPoint: Number(changes.reorderPoint ?? product.reorderPoint),
              updatedAt: new Date().toISOString(),
            }
          : product
      ),
    }));
  };

  const adjustInventory = (productId, quantity, reason = "Manual admin adjustment") => {
    const movement = {
      id: crypto.randomUUID(),
      productId,
      type: quantity >= 0 ? "restock" : "adjustment",
      quantity: Number(quantity),
      reason,
      createdAt: new Date().toISOString(),
    };
    setState((current) => ({
      ...current,
      products: current.products.map((product) =>
        String(product.id) === String(productId)
          ? { ...product, stock: Math.max(0, Number(product.stock) + Number(quantity)), updatedAt: new Date().toISOString() }
          : product
      ),
      inventoryMovements: [movement, ...current.inventoryMovements],
    }));
  };

  const createProduct = (input) => {
    const product = makeProduct(input);
    setState((current) => ({ ...current, products: [product, ...current.products] }));
    return product;
  };

  const createUser = (input) => {
    const user = makeUser(input);
    setState((current) => {
      const exists = current.users.some((candidate) => candidate.email.toLowerCase() === user.email);
      if (exists) return current;
      return { ...current, users: [user, ...current.users] };
    });
    return user;
  };

  const updateUser = (userId, changes) => {
    setState((current) => ({
      ...current,
      users: current.users.map((candidate) =>
        candidate.id === userId ? { ...candidate, ...changes, updatedAt: new Date().toISOString() } : candidate
      ),
    }));
  };

  const submitSellerApplication = ({ user, form }) => {
    const application = makeSellerApplication({ user, form });
    setState((current) => ({
      ...current,
      sellerApplications: [application, ...(current.sellerApplications || [])],
      users: current.users.map((candidate) =>
        candidate.id === user.id ? { ...candidate, role: roles.seller } : candidate
      ),
    }));
    return application;
  };

  const reviewSellerApplication = ({ applicationId, status, reviewer }) => {
    setState((current) => {
      const application = (current.sellerApplications || []).find((item) => item.id === applicationId);
      if (!application) return current;
      const approved = status === "approved";
      const sellerId = approved ? application.sellerId || `seller_${crypto.randomUUID()}` : application.sellerId;
      const seller = approved
        ? {
            id: sellerId,
            userId: application.userId,
            companyName: application.companyName,
            publicName: application.publicName || application.companyName,
            legalName: application.legalName,
            taxId: application.taxId,
            status: "approved",
            riskStatus: "reviewed",
            categories: application.categories,
            productDescription: application.productDescription,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          }
        : null;

      return {
        ...current,
        sellerApplications: current.sellerApplications.map((item) =>
          item.id === applicationId
            ? {
                ...item,
                sellerId,
                status,
                riskStatus: approved ? "reviewed" : "blocked",
                reviewedAt: new Date().toISOString(),
                reviewedBy: reviewer?.id || "system",
                updatedAt: new Date().toISOString(),
              }
            : item
        ),
        sellers: approved && !current.sellers.some((item) => item.id === sellerId)
          ? [seller, ...current.sellers]
          : current.sellers,
        users: approved
          ? current.users.map((candidate) =>
              candidate.id === application.userId
                ? { ...candidate, role: roles.seller, sellerId }
                : candidate
            )
          : current.users,
      };
    });
  };

  const deleteProduct = (productId) => {
    setState((current) => ({
      ...current,
      products: current.products.filter((product) => String(product.id) !== String(productId)),
    }));
  };

  const createCategory = (input) => {
    const category = makeCategory(input);
    setState((current) => ({ ...current, categories: [category, ...current.categories] }));
    return category;
  };

  const updateCategory = (categoryId, changes) => {
    setState((current) => ({
      ...current,
      categories: current.categories.map((category) =>
        category.id === categoryId ? { ...category, ...changes, updatedAt: new Date().toISOString() } : category
      ),
    }));
  };

  const deleteCategory = (categoryId) => {
    setState((current) => {
      const hasProducts = current.products.some((product) => product.categoryId === categoryId);
      if (hasProducts) return current;
      return { ...current, categories: current.categories.filter((category) => category.id !== categoryId) };
    });
  };

  const createOrder = ({ user, items, shipping, paymentStatus, status, shippingAddress }) => {
    const order = makeOrder({ user, items, shipping, paymentStatus, status, shippingAddress });
    setState((current) => ({ ...current, orders: [order, ...current.orders] }));
    return order;
  };

  const updateOrder = (orderId, changes, note = "Admin update.") => {
    setState((current) => ({
      ...current,
      orders: current.orders.map((order) =>
        order.id === orderId
          ? {
              ...order,
              ...changes,
              timeline: changes.status
                ? [...order.timeline, { status: changes.status, at: new Date().toISOString(), note }]
                : order.timeline,
              updatedAt: new Date().toISOString(),
            }
          : order
      ),
    }));
  };

  const recordPayment = ({ order, provider, providerTransactionId, status }) => {
    const payment = makePayment({ order, provider, providerTransactionId, status });
    setState((current) => ({
      ...current,
      payments: [payment, ...current.payments],
      products:
        status === "succeeded"
          ? current.products.map((product) => {
              const line = order.items.find((item) => String(item.productId) === String(product.id));
              return line
                ? { ...product, stock: Math.max(0, Number(product.stock) - Number(line.quantity)), updatedAt: new Date().toISOString() }
                : product;
            })
          : current.products,
      inventoryMovements:
        status === "succeeded"
          ? [
              ...order.items.map((item) => ({
                id: crypto.randomUUID(),
                productId: item.productId,
                type: "sale",
                quantity: -Number(item.quantity),
                reason: `Order ${order.orderNumber}`,
                createdAt: new Date().toISOString(),
              })),
              ...current.inventoryMovements,
            ]
          : current.inventoryMovements,
      orders: current.orders.map((candidate) =>
        candidate.id === order.id
          ? {
              ...candidate,
              paymentStatus: status,
              status: status === "succeeded" ? "paid" : candidate.status,
              updatedAt: new Date().toISOString(),
            }
          : candidate
      ),
    }));
    return payment;
  };

  const updatePayment = (paymentId, changes) => {
    setState((current) => {
      const payment = current.payments.find((item) => item.id === paymentId);
      return {
        ...current,
        payments: current.payments.map((item) =>
          item.id === paymentId ? { ...item, ...changes, updatedAt: new Date().toISOString() } : item
        ),
        orders: payment
          ? current.orders.map((order) =>
              order.id === payment.orderId
                ? {
                    ...order,
                    paymentStatus: changes.status || order.paymentStatus,
                    status: changes.status === "refunded" ? "refunded" : order.status,
                    updatedAt: new Date().toISOString(),
                  }
                : order
            )
          : current.orders,
      };
    });
  };

  const value = useMemo(
    () => ({
      ...state,
      analytics: getAnalytics(state),
      createUser,
      updateUser,
      createProduct,
      updateProduct,
      adjustInventory,
      deleteProduct,
      createCategory,
      updateCategory,
      deleteCategory,
      createOrder,
      updateOrder,
      recordPayment,
      updatePayment,
      submitSellerApplication,
      reviewSellerApplication,
      getSellerById: (sellerId) => state.sellers?.find((seller) => seller.id === sellerId),
    }),
    [state]
  );

  return <CommerceContext.Provider value={value}>{children}</CommerceContext.Provider>;
}

export function useCommerce() {
  const context = useContext(CommerceContext);
  if (!context) throw new Error("useCommerce must be used within CommerceProvider");
  return context;
}
