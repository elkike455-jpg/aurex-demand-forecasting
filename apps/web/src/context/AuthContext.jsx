import { createContext, useContext, useMemo, useState } from "react";

const AUTH_STORAGE_KEY = "aurex-user-session";
const AuthContext = createContext(null);

function loadSession() {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(loadSession);

  const saveUser = (nextUser) => {
    setUser(nextUser);
    if (!nextUser) {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextUser));
  };

  const signIn = ({ email, fullName }) => {
    const nextUser = {
      id: crypto.randomUUID(),
      email,
      fullName,
    };
    saveUser(nextUser);
    return nextUser;
  };

  const signOut = () => saveUser(null);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      signIn,
      signOut,
    }),
    [user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
