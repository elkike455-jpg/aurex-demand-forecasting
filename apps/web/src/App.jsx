import { useEffect, useState } from "react";

export default function App() {
  const [msg, setMsg] = useState("Loading...");

  useEffect(() => {
    setMsg("✅ Aurex Web is running (Vite + React).");
  }, []);

  return (
    <div style={{ padding: 24, fontFamily: "system-ui" }}>
      <h1>AUREX – Demand Forecasting</h1>
      <p>{msg}</p>
      <p>Next step: connect to FastAPI <code>/health</code>.</p>
    </div>
  );
}
