export const mockProducts = [
  {
    id: 1,
    name: "Wireless Bluetooth Earbuds Pro",
    price: 129.99,
    oldPrice: 169.99,
    image:
      "https://images.pexels.com/photos/373945/pexels-photo-373945.jpeg?auto=compress&cs=tinysrgb&w=600",
    rating: 4.7,
    description:
      "Premium sound quality with active noise cancellation. Up to 24h battery life, IPX5 water resistance, and a sleek charging case. Perfect for work and workouts.",
  },
  {
    id: 2,
    name: "Smartphone 128GB 6.5\" Display",
    price: 699.0,
    image:
      "https://images.pexels.com/photos/404280/pexels-photo-404280.jpeg?auto=compress&cs=tinysrgb&w=600",
    rating: 4.5,
    description:
      "Fast performance, stunning AMOLED display, and a versatile triple camera. 128GB storage, 5G ready, and all-day battery. Your next daily driver.",
  },
  {
    id: 3,
    name: "Running sneakers – midnight edition",
    price: 89.99,
    oldPrice: 119.99,
    image:
      "https://images.pexels.com/photos/19090/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=600",
    rating: 4.3,
    description:
      "Lightweight, breathable design with responsive cushioning. Durable outsole for road and light trail. Style that works from the track to the street.",
  },
  {
    id: 4,
    name: "LED desk lamp with wireless charger",
    price: 59.99,
    image:
      "https://images.pexels.com/photos/8091470/pexels-photo-8091470.jpeg?auto=compress&cs=tinysrgb&w=600",
    rating: 4.6,
    description:
      "Adjustable brightness and color temperature. Built-in Qi wireless charger for your phone. Minimal design that fits any desk or bedside table.",
  },
];

export function getProductById(id) {
  const numId = Number(id);
  return mockProducts.find((p) => p.id === numId) ?? null;
}

export function getRelatedProducts(productId, limit = 4) {
  const numId = Number(productId);
  return mockProducts.filter((p) => p.id !== numId).slice(0, limit);
}

export const mockReviews = [
  { id: 1, productId: 1, author: "Alex M.", rating: 5, text: "Best earbuds I've owned. ANC is incredible and battery lasts all week.", date: "2 days ago" },
  { id: 2, productId: 1, author: "Jordan K.", rating: 4, text: "Great sound and fit. Only minus is the case is a bit bulky.", date: "1 week ago" },
  { id: 3, productId: 2, author: "Sam R.", rating: 5, text: "Fast, great camera, and the display is stunning. Worth every penny.", date: "3 days ago" },
  { id: 4, productId: 3, author: "Casey L.", rating: 4, text: "Very comfortable for long runs. True to size.", date: "5 days ago" },
  { id: 5, productId: 4, author: "Taylor W.", rating: 5, text: "Love the wireless charging built in. Clean look on my desk.", date: "1 week ago" },
];

export function getReviewsByProductId(productId) {
  return mockReviews.filter((r) => r.productId === Number(productId));
}

