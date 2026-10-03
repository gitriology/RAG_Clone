import {
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
} from "firebase/auth";
import { auth, isConfigured } from "./config";

const requireFirebase = () => {
  if (!isConfigured || !auth) {
    throw new Error(
      "Firebase is not configured. Add the VITE_FIREBASE_* values to .env.",
    );
  }
};

export const signInWithGoogle = async () => {
  requireFirebase();
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
  return signInWithPopup(auth, provider);
};

export const logout = async () => {
  requireFirebase();
  return signOut(auth);
};

export const subscribeToAuthChanges = (callback) => {
  if (!isConfigured || !auth) return () => {};
  return onAuthStateChanged(auth, callback);
};
