import { BrowserRouter, Routes, Route } from "react-router-dom";
import { CartProvider } from "./context/CartContext";
import { AuthProvider } from "./context/AuthContext";
import { AppLayout } from "./layouts/AppLayout";
import { HomePage } from "./pages/HomePage";
import { ProductListPage } from "./pages/ProductListPage";
import { ProductDetailPage } from "./pages/ProductDetailPage";
import { CartPage } from "./pages/CartPage";
import { CheckoutPage } from "./pages/CheckoutPage";
import { SignInPage } from "./pages/SignInPage";
import { CreateAccountPage } from "./pages/CreateAccountPage";
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
      <AuthProvider>
        <CartProvider>
          <AppLayout>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/products" element={<ProductListPage />} />
              <Route path="/product/:id" element={<ProductDetailPage />} />
              <Route path="/cart" element={<CartPage />} />
              <Route path="/checkout" element={<CheckoutPage />} />
              <Route path="/sign-in" element={<SignInPage />} />
              <Route path="/create-account" element={<CreateAccountPage />} />
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
        </CartProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
