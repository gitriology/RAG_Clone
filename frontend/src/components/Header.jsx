import { ChevronDown, LogOut, Menu, Moon, Sun } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import LogoMark from "./LogoMark";

function Header({ onMenuClick, theme, onToggleTheme }) {
  const { user, signOut } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  const initial = (user?.displayName || user?.email || "S")
    .charAt(0)
    .toUpperCase();

  return (
    <header className="topbar">
      <button
        type="button"
        className="mobile-menu-btn"
        onClick={onMenuClick}
        aria-label="Open sidebar"
      >
        <Menu size={20} />
      </button>
      <div className="mobile-brand">
        <LogoMark className="mobile-logo" />
        <span>RAGar</span>
      </div>
      <div className="topbar-actions">
        <button
          className="icon-button"
          onClick={onToggleTheme}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun size={19} /> : <Moon size={19} />}
        </button>
        <div className="user-menu-wrap">
          <button
            className="user-button"
            onClick={() => setMenuOpen((v) => !v)}
          >
            {user?.photoURL ? (
              <img src={user.photoURL} alt="" className="avatar-image" />
            ) : (
              <span className="avatar">{initial}</span>
            )}
            <span className="user-name">
              {user?.displayName || "Researcher"}
            </span>
            <ChevronDown size={15} />
          </button>
          {menuOpen && (
            <div className="user-menu">
              <div className="user-menu-meta">
                <strong>{user?.displayName || "Researcher"}</strong>
                <span>{user?.email}</span>
              </div>
              <button onClick={signOut}>
                <LogOut size={16} /> Log out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
