import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

interface User {
  id: string;
  email: string;
  role: string;
}

interface AuthContextValue {
  user: User | null;
  accessToken: string | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  requestPasswordReset: (email: string) => Promise<void>;
  resetPassword: (token: string, password: string) => Promise<void>;
  verifyEmail: (token: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:3001";
export const authEnabled = import.meta.env.VITE_ENABLE_AUTH === "true";
const anonymousAuth: AuthContextValue = {
  user: null,
  accessToken: null,
  loading: false,
  signIn: async () => undefined,
  signUp: async () => undefined,
  requestPasswordReset: async () => undefined,
  resetPassword: async () => undefined,
  verifyEmail: async () => undefined,
  signOut: async () => undefined,
};

async function requestAuth(path: string, body?: object) {
  const response = await fetch(`${backendUrl}${path}`, {
    method: "POST",
    credentials: "include",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(payload?.error || "Authentication request failed");
  return payload;
}

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authEnabled) {
      setLoading(false);
      return;
    }
    requestAuth("/auth/refresh")
      .then((payload) => {
        setUser(payload.user);
        setAccessToken(payload.accessToken);
      })
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    accessToken,
    loading,
    async signIn(email, password) {
      const payload = await requestAuth("/auth/login", { email, password });
      setUser(payload.user);
      setAccessToken(payload.accessToken);
    },
    async signUp(email, password) {
      const payload = await requestAuth("/auth/register", { email, password });
      setUser(payload.user);
      setAccessToken(payload.accessToken);
    },
    async requestPasswordReset(email) {
      await requestAuth("/auth/request-password-reset", { email });
    },
    async resetPassword(token, password) {
      await requestAuth("/auth/reset-password", { token, password });
    },
    async verifyEmail(token) {
      await requestAuth("/auth/verify-email", { token });
    },
    async signOut() {
      await requestAuth("/auth/logout");
      setUser(null);
      setAccessToken(null);
    },
  }), [accessToken, loading, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export function useAuth() {
  return useContext(AuthContext) ?? anonymousAuth;
}
