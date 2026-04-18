import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { en } from "../i18n/en";
import { es } from "../i18n/es";

const STORAGE_KEY = "aurex-language";
const translations = { en, es };

function getInitialLanguage() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "en" || stored === "es") return stored;
  return window.navigator.language?.toLowerCase().startsWith("es") ? "es" : "en";
}

function getValue(source, path) {
  return path.split(".").reduce((value, key) => value?.[key], source);
}

function interpolate(value, params = {}) {
  if (typeof value !== "string") return value;
  const cleanValue = repairMojibake(value);
  return Object.entries(params).reduce(
    (text, [key, replacement]) => text.replaceAll(`{{${key}}}`, replacement),
    cleanValue
  );
}

function repairMojibake(value) {
  if (!/[ÃÂ]/.test(value)) return value;
  try {
    return decodeURIComponent(escape(value));
  } catch {
    return value;
  }
}

function translateCollection(value) {
  if (typeof value === "string") return repairMojibake(value);
  if (Array.isArray(value)) return value.map(translateCollection);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, translateCollection(item)])
    );
  }
  return value;
}

const LanguageContext = createContext(null);

export function LanguageProvider({ children }) {
  const [language, setLanguageState] = useState(getInitialLanguage);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, language);
    document.documentElement.lang = language;
  }, [language]);

  const value = useMemo(() => {
    const dictionary = translations[language] ?? en;

    const t = (path, params) => {
      const found = getValue(dictionary, path) ?? getValue(en, path) ?? path;
      if (params === undefined && typeof found !== "string") {
        return translateCollection(found);
      }
      return interpolate(found, params);
    };

    const setLanguage = (nextLanguage) => {
      if (nextLanguage === "en" || nextLanguage === "es") {
        setLanguageState(nextLanguage);
      }
    };

    const translateProduct = (product) => ({
      ...product,
      name: repairMojibake(
        getValue(dictionary, `products.names.${product.id}`) ||
          getValue(en, `products.names.${product.id}`) ||
          product.name
      ),
      description: repairMojibake(
        getValue(dictionary, `products.descriptions.${product.id}`) ||
          getValue(en, `products.descriptions.${product.id}`) ||
          product.description
      ),
    });

    const translateReview = (review) => ({
      ...review,
      text: t(`reviews.text.${review.id}`),
      date: t(`reviews.dates.${review.date}`),
    });

    return {
      language,
      languages: ["en", "es"],
      setLanguage,
      t,
      translateProduct,
      translateReview,
      dictionary,
    };
  }, [language]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within LanguageProvider");
  }
  return context;
}

