import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import { ChevronRight } from "lucide-react";

const breadcrumbMap: Record<string, string> = {
  overview: "Overview",
  runs: "Runs",
  artifacts: "Artifacts",
  loops: "Loops",
  prompts: "Prompts",
  providers: "Providers",
  settings: "Settings",
};

export default function Layout() {
  const location = useLocation();
  const segments = location.pathname.split("/").filter(Boolean);

  return (
    <div className="min-h-screen bg-surface">
      <Sidebar />
      <div className="ml-60">
        {/* Header */}
        <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-surface-border px-8 py-4">
          <nav className="flex items-center gap-1 text-sm text-gray-500">
            <span className="text-gray-400">MI-Workbench</span>
            {segments.map((seg, i) => (
              <span key={i} className="flex items-center gap-1">
                <ChevronRight className="w-4 h-4 text-gray-300" />
                <span
                  className={
                    i === segments.length - 1
                      ? "text-gray-800 font-medium"
                      : ""
                  }
                >
                  {breadcrumbMap[seg] || seg}
                </span>
              </span>
            ))}
          </nav>
        </header>

        {/* Main content */}
        <main className="p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
