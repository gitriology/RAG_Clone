import { Chrome, Sparkles } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import LogoMark from "./LogoMark";

function Login() {
  const { configured, signIn, authError } = useAuth();

  return (
    <main className="login-page">
      <div className="login-glow login-glow-one" />
      <div className="login-glow login-glow-two" />
      <section className="login-card">
        <div className="login-brand">
          <LogoMark className="login-logo" />
          <div>
            <div className="brand-wordmark">RAGar</div>
            <div className="brand-tagline">Search · Reason · Generate</div>
          </div>
        </div>

        <div className="login-icon">
          <Sparkles size={18} />
        </div>
        <h1>Research, retrieved.</h1>
        <p className="login-description">
          A grounded RAG workspace for exploring answers, sources, evidence
          graphs, and retrieval state in one place.
        </p>

        <button
          className="google-button"
          onClick={signIn}
          disabled={!configured}
        >
          <Chrome size={18} />
          Continue with Google
        </button>

        {!configured && (
          <div className="setup-notice">
            <strong>Firebase setup required</strong>
            <span>
              Add the VITE_FIREBASE_* values to <code>frontend/.env</code> after
              creating your Firebase web app.
            </span>
          </div>
        )}

        {authError && <div className="auth-error">{authError}</div>}

        <p className="login-footnote">
          Your account controls access to your private conversation history.
        </p>
      </section>
    </main>
  );
}

export default Login;
