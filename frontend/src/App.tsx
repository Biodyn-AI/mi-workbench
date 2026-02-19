import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Runs from "./pages/Runs";
import RunDetail from "./pages/RunDetail";
import Artifacts from "./pages/Artifacts";
import Loops from "./pages/Loops";
import Prompts from "./pages/Prompts";
import Providers from "./pages/Providers";
import Settings from "./pages/Settings";
import Telemetry from "./pages/Telemetry";
import Knowledge from "./pages/Knowledge";
import Comparison from "./pages/Comparison";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/overview" replace />} />
        <Route path="/overview" element={<Overview />} />
        <Route path="/runs" element={<Runs />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="/artifacts" element={<Artifacts />} />
        <Route path="/loops" element={<Loops />} />
        <Route path="/prompts" element={<Prompts />} />
        <Route path="/providers" element={<Providers />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/telemetry" element={<Telemetry />} />
        <Route path="/knowledge" element={<Knowledge />} />
        <Route path="/comparison" element={<Comparison />} />
      </Route>
    </Routes>
  );
}
