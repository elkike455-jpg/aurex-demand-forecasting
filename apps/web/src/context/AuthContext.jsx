import { createContext, useContext, useMemo, useState } from "react";
import { GENERAL_ADMIN_EMAIL, loadCommerceState, roles, seedState } from "../data/aurexStore";

const AUTH_STORAGE_KEY = "aurex-user-session";
const AUTH_ACCOUNTS_KEY = "aurex-local-accounts";
const AUTH_SECURITY_KEY = "aurex-auth-security";
const AUTH_CHALLENGE_KEY = "aurex-auth-challenge";
const AuthContext = createContext(null);

function loadSession() {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    const session = raw ? JSON.parse(raw) : null;
    return session?.email?.toLowerCase() === GENERAL_ADMIN_EMAIL
      ? { ...session, role: roles.superadmin, sellerId: session.sellerId || "seller_aurex" }
      : session;
  } catch {
    return null;
  }
}

function publicUser(user) {
  return {
    id: user.id,
    email: user.email,
    fullName: user.fullName,
    role: user.role,
    sellerId: user.sellerId || null,
  };
}

function readJson(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function getAllUsers() {
  const commerceUsers = loadCommerceState().users || [];
  const localUsers = readJson(AUTH_ACCOUNTS_KEY, []);
  const byEmail = new Map();
  [...seedState.users, ...commerceUsers, ...localUsers].forEach((candidate) => {
    if (!candidate?.email) return;
    const email = candidate.email.toLowerCase();
    byEmail.set(
      email,
      email === GENERAL_ADMIN_EMAIL
        ? { ...candidate, role: roles.superadmin, sellerId: candidate.sellerId || "seller_aurex" }
        : candidate
    );
  });
  return [...byEmail.values()];
}

function generateCode() {
  return String(Math.floor(100000 + Math.random() * 900000));
}

function getSecurityState() {
  return readJson(AUTH_SECURITY_KEY, {});
}

function saveSecurityState(next) {
  window.localStorage.setItem(AUTH_SECURITY_KEY, JSON.stringify(next));
}

function loadChallenge() {
  const challenge = readJson(AUTH_CHALLENGE_KEY, null);
  if (!challenge) return null;
  if (Date.now() > challenge.expiresAt) {
    window.localStorage.removeItem(AUTH_CHALLENGE_KEY);
    return null;
  }
  return challenge;
}

function saveChallenge(challenge) {
  if (!challenge) {
    window.localStorage.removeItem(AUTH_CHALLENGE_KEY);
    return;
  }
  window.localStorage.setItem(AUTH_CHALLENGE_KEY, JSON.stringify(challenge));
}

function registerLocalAccount(account) {
  const accounts = readJson(AUTH_ACCOUNTS_KEY, []);
  const filtered = accounts.filter((item) => item.email.toLowerCase() !== account.email.toLowerCase());
  window.localStorage.setItem(AUTH_ACCOUNTS_KEY, JSON.stringify([account, ...filtered]));
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(loadSession);
  const [pendingChallenge, setPendingChallenge] = useState(loadChallenge);

  const saveUser = (nextUser) => {
    setUser(nextUser);
    if (!nextUser) {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextUser));
  };

  const registerAccount = (account) => {
    registerLocalAccount(account);
    const nextUser = publicUser(account);
    saveUser(nextUser);
    return nextUser;
  };

  const updateCurrentUser = (changes) => {
    if (!user) return null;
    const nextUser = { ...user, ...changes };
    saveUser(nextUser);
    const accounts = readJson(AUTH_ACCOUNTS_KEY, []);
    const updatedAccounts = accounts.map((account) =>
      account.id === user.id ? { ...account, ...changes } : account
    );
    window.localStorage.setItem(AUTH_ACCOUNTS_KEY, JSON.stringify(updatedAccounts));
    return nextUser;
  };

  const signIn = ({ email, fullName, password }) => {
    const normalizedEmail = email.trim().toLowerCase();
    const security = getSecurityState();
    const accountSecurity = security[normalizedEmail] || { attempts: 0, lockedUntil: 0 };
    if (accountSecurity.lockedUntil && Date.now() < accountSecurity.lockedUntil) {
      throw new Error("Account locked temporarily. Try again in one minute.");
    }

    const seedUser = getAllUsers().find(
      (candidate) => candidate.email.toLowerCase() === email.toLowerCase()
    );

    if (seedUser) {
      if (password && seedUser.password !== password) {
        const attempts = (accountSecurity.attempts || 0) + 1;
        security[normalizedEmail] = {
          attempts,
          lockedUntil: attempts >= 5 ? Date.now() + 60_000 : 0,
        };
        saveSecurityState(security);
        throw new Error("Invalid credentials");
      }
      security[normalizedEmail] = { attempts: 0, lockedUntil: 0 };
      saveSecurityState(security);
      const code = generateCode();
      const challenge = {
        user: publicUser(seedUser),
        code,
        expiresAt: Date.now() + 5 * 60_000,
      };
      setPendingChallenge(challenge);
      saveChallenge(challenge);
      return { requiresCode: true, demoCode: code, expiresAt: challenge.expiresAt };
    }

    throw new Error(`No account exists for ${fullName || normalizedEmail}. Create an account first.`);
  };

  const verifySignInCode = (code) => {
    const activeChallenge = pendingChallenge || loadChallenge();
    if (!activeChallenge) throw new Error("No verification challenge is active.");
    if (Date.now() > activeChallenge.expiresAt) {
      setPendingChallenge(null);
      saveChallenge(null);
      throw new Error("Verification code expired.");
    }
    if (String(code).trim() !== activeChallenge.code) {
      throw new Error("Invalid verification code.");
    }
    saveUser(activeChallenge.user);
    setPendingChallenge(null);
    saveChallenge(null);
    return activeChallenge.user;
  };

  const signOut = () => {
    setPendingChallenge(null);
    saveChallenge(null);
    saveUser(null);
  };

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      isAdmin: user?.role === roles.superadmin || user?.role === roles.admin,
      isStaff: user?.role === roles.superadmin || user?.role === roles.admin || user?.role === roles.staff,
      isSeller: user?.role === roles.superadmin || user?.role === roles.seller,
      pendingChallenge,
      registerAccount,
      updateCurrentUser,
      signIn,
      verifySignInCode,
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
