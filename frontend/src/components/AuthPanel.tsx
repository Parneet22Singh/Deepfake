import { useEffect, useState } from "react";
import { authEnabled, useAuth } from "@/contexts/AuthContext";

export const AuthPanel = () => {
  if (!authEnabled) return null;
  return <EnabledAuthPanel />;
};

const EnabledAuthPanel = () => {
  const { user, loading, signIn, signUp, signOut, requestPasswordReset, resetPassword, verifyEmail } = useAuth();
  const [mode, setMode] = useState<"login" | "register" | "forgot" | "reset">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const resetToken = new URLSearchParams(window.location.search).get("reset_token") || "";
  const verificationToken = new URLSearchParams(window.location.search).get("verify_email") || "";

  useEffect(() => {
    if (resetToken) setMode("reset");
    if (verificationToken) void verifyEmail(verificationToken);
  }, [resetToken, verificationToken, verifyEmail]);

  if (loading) return null;
  if (user) {
    return (
      <div className="fixed right-4 top-4 z-30 flex items-center gap-3 rounded-lg border border-theta/30 bg-background/90 px-3 py-2 text-xs font-mono backdrop-blur">
        <span className="text-theta">{user.email}</span>
        <button onClick={() => void signOut()} className="text-muted-foreground hover:text-foreground">SIGN OUT</button>
      </div>
    );
  }

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") await signUp(email, password);
      else if (mode === "forgot") await requestPasswordReset(email);
      else if (mode === "reset") await resetPassword(resetToken, password);
      else await signIn(email, password);
      if (mode === "forgot") setError("If an account exists, a reset link has been sent.");
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed right-4 top-4 z-30 w-72 rounded-lg border border-p300/30 bg-background/95 p-4 shadow-lg backdrop-blur">
      <p className="mb-3 font-display text-xs tracking-widest text-p300">{mode === "register" ? "CREATE ACCOUNT" : mode === "forgot" ? "ACCOUNT RECOVERY" : mode === "reset" ? "RESET PASSWORD" : "ACCOUNT LOGIN"}</p>
      {mode !== "reset" && <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="Email" className="mb-2 w-full rounded border border-border bg-muted px-3 py-2 text-sm outline-none" />}
      {mode !== "forgot" && <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="Password (12+ characters)" className="mb-2 w-full rounded border border-border bg-muted px-3 py-2 text-sm outline-none" />}
      {error && <p className="mb-2 text-xs text-shred">{error}</p>}
      <button disabled={busy} onClick={() => void submit()} className="w-full rounded bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-50">
        {busy ? "PROCESSING..." : mode === "register" ? "REGISTER" : mode === "forgot" ? "SEND RESET LINK" : mode === "reset" ? "RESET PASSWORD" : "SIGN IN"}
      </button>
      <button onClick={() => { setMode(mode === "register" ? "login" : "register"); setError(null); }} className="mt-2 w-full text-xs text-muted-foreground hover:text-foreground">
        {mode === "register" ? "Already have an account? Sign in" : "Need an account? Register"}
      </button>
      {mode === "login" && <button onClick={() => setMode("forgot")} className="mt-1 w-full text-xs text-muted-foreground hover:text-foreground">Forgot password?</button>}
    </div>
  );
};
