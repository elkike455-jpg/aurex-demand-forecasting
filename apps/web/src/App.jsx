import { BrowserRouter, Routes, Route } from "react-router-dom";
import { CartProvider } from "./context/CartContext";
import { AuthProvider } from "./context/AuthContext";
import { LanguageProvider } from "./context/LanguageContext";
import { CommerceProvider } from "./context/CommerceContext";
import { ProtectedRoute } from "./routes/ProtectedRoute";
import { AppLayout } from "./layouts/AppLayout";
import { AdminLayout } from "./admin/components/AdminLayout";
import { AdminDashboardPage } from "./admin/pages/AdminDashboardPage";
import { AdminProductsPage } from "./admin/pages/AdminProductsPage";
import { AdminProductDetailPage } from "./admin/pages/AdminProductDetailPage";
import { AdminCategoriesPage } from "./admin/pages/AdminCategoriesPage";
import { AdminOrdersPage } from "./admin/pages/AdminOrdersPage";
import { AdminOrderDetailPage } from "./admin/pages/AdminOrderDetailPage";
import { AdminUsersPage } from "./admin/pages/AdminUsersPage";
import { AdminPaymentsPage } from "./admin/pages/AdminPaymentsPage";
import { AdminInventoryPage } from "./admin/pages/AdminInventoryPage";
import { AdminAnalyticsPage } from "./admin/pages/AdminAnalyticsPage";
import { AdminSettingsPage } from "./admin/pages/AdminSettingsPage";
import { AdminVendorsPage } from "./admin/pages/AdminVendorsPage";
import { HomePage } from "./pages/HomePage";
import { ProductListPage } from "./pages/ProductListPage";
import { CategoriesPage } from "./pages/CategoriesPage";
import { CategoryDetailPage } from "./pages/CategoryDetailPage";
import { ProductDetailPage } from "./pages/ProductDetailPage";
import { CartPage } from "./pages/CartPage";
import { CheckoutPage } from "./pages/CheckoutPage";
import { ReceiptPage } from "./pages/ReceiptPage";
import { SignInPage } from "./pages/SignInPage";
import { CreateAccountPage } from "./pages/CreateAccountPage";
import { AccountPage } from "./pages/AccountPage";
import { SellerOnboardingPage } from "./pages/SellerOnboardingPage";
import { SellerDashboardPage } from "./pages/SellerDashboardPage";
import { AboutPage } from "./pages/AboutPage";
import { ContactPage } from "./pages/ContactPage";
import { FAQPage } from "./pages/FAQPage";
import { ShippingReturnsPage } from "./pages/ShippingReturnsPage";
import { PrivacyPage } from "./pages/PrivacyPage";
import { TermsPage } from "./pages/TermsPage";
import { TrackOrderPage } from "./pages/TrackOrderPage";
import { BlogPage } from "./pages/BlogPage";
import { CareersPage } from "./pages/CareersPage";
import { StoreLocatorPage } from "./pages/StoreLocatorPage";
import { GiftCardsPage } from "./pages/GiftCardsPage";

export default function App() {
  return (
    <BrowserRouter>
      <LanguageProvider>
        <AuthProvider>
          <CommerceProvider>
            <CartProvider>
              <Routes>
                <Route
                  path="/admin"
                  element={
                    <ProtectedRoute allowedRoles={["superadmin", "admin", "staff"]}>
                      <AdminLayout />
                    </ProtectedRoute>
                  }
                >
                  <Route index element={<AdminDashboardPage />} />
                  <Route path="products" element={<AdminProductsPage />} />
                  <Route path="products/:id" element={<AdminProductDetailPage />} />
                  <Route path="categories" element={<AdminCategoriesPage />} />
                  <Route path="orders" element={<AdminOrdersPage />} />
                  <Route path="orders/:id" element={<AdminOrderDetailPage />} />
                  <Route path="customers" element={<AdminUsersPage />} />
                  <Route path="vendors" element={<AdminVendorsPage />} />
                  <Route path="payments" element={<AdminPaymentsPage />} />
                  <Route path="inventory" element={<AdminInventoryPage />} />
                  <Route path="analytics" element={<AdminAnalyticsPage />} />
                  <Route path="settings" element={<AdminSettingsPage />} />
                </Route>
                <Route
                  path="*"
                  element={
                    <AppLayout>
                      <Routes>
                        <Route path="/" element={<HomePage />} />
                        <Route path="/products" element={<ProductListPage />} />
                        <Route path="/categories" element={<CategoriesPage />} />
                        <Route path="/category/:slug" element={<CategoryDetailPage />} />
                        <Route path="/product/:id" element={<ProductDetailPage />} />
                        <Route path="/cart" element={<CartPage />} />
                        <Route path="/checkout" element={<CheckoutPage />} />
                        <Route path="/receipt/:id" element={<ReceiptPage />} />
                        <Route path="/sign-in" element={<SignInPage />} />
                        <Route path="/create-account" element={<CreateAccountPage />} />
                        <Route
                          path="/account"
                          element={
                            <ProtectedRoute allowedRoles={["superadmin", "admin", "staff", "seller", "customer"]}>
                              <AccountPage />
                            </ProtectedRoute>
                          }
                        />
                        <Route path="/seller/onboarding" element={<SellerOnboardingPage />} />
                        <Route
                          path="/seller/dashboard"
                          element={
                            <ProtectedRoute allowedRoles={["superadmin", "admin", "seller"]}>
                              <SellerDashboardPage />
                            </ProtectedRoute>
                          }
                        />
                        <Route path="/about" element={<AboutPage />} />
                        <Route path="/contact" element={<ContactPage />} />
                        <Route path="/help" element={<FAQPage />} />
                        <Route path="/shipping-returns" element={<ShippingReturnsPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="/track-order" element={<TrackOrderPage />} />
                        <Route path="/blog" element={<BlogPage />} />
                        <Route path="/careers" element={<CareersPage />} />
                        <Route path="/stores" element={<StoreLocatorPage />} />
                        <Route path="/gift-cards" element={<GiftCardsPage />} />
                      </Routes>
                    </AppLayout>
                  }
                />
              </Routes>
            </CartProvider>
          </CommerceProvider>
        </AuthProvider>
      </LanguageProvider>
    </BrowserRouter>
  );
}

