import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  logout,
  signInWithGoogle,
  subscribeToAuthChanges,
} from "../firebase/auth";
import { isConfigured } from "../firebase/config";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(isConfigured);
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    if (!isConfigured) return;

    const unsubscribe = subscribeToAuthChanges((nextUser) => {
      setUser(nextUser);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      configured: isConfigured,
      authError,
      clearAuthError: () => setAuthError(""),
      signIn: async () => {
        setAuthError("");
        try {
          await signInWithGoogle();
        } catch (error) {
          if (error?.code === "auth/popup-closed-by-user") return;
          setAuthError(
            error?.message || "Google sign-in failed. Please try again.",
          );
        }
      },
      signOut: async () => {
        setAuthError("");
        try {
          await logout();
        } catch (error) {
          setAuthError(error?.message || "Sign out failed. Please try again.");
        }
      },
    }),
    [user, loading, authError],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
