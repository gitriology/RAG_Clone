import darkMark from "../assets/branding/ragar-mark-dark.png";
import lightMark from "../assets/branding/ragar-mark-light.png";

function LogoMark({ className = "" }) {
  return (
    <>
      <img
        className={`logo-mark-image logo-mark-dark ${className}`}
        src={darkMark}
        alt="RAGar"
      />

      <img
        className={`logo-mark-image logo-mark-light ${className}`}
        src={lightMark}
        alt="RAGar"
      />
    </>
  );
}

export default LogoMark;
