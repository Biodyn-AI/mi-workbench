import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Play,
  FileText,
  GitFork,
  MessageSquare,
  Server,
  Settings,
  Microscope,
  Activity,
  Brain,
  GitCompare,
} from "lucide-react";

const navItems = [
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
  { to: "/runs", label: "Runs", icon: Play },
  { to: "/comparison", label: "Compare", icon: GitCompare },
  { to: "/artifacts", label: "Artifacts", icon: FileText },
  { to: "/loops", label: "Loops", icon: GitFork },
  { to: "/prompts", label: "Prompts", icon: MessageSquare },
  { to: "/providers", label: "Providers", icon: Server },
  { to: "/telemetry", label: "Telemetry", icon: Activity },
  { to: "/knowledge", label: "Knowledge", icon: Brain },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 bottom-0 w-60 bg-sidebar flex flex-col z-20">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-white/10">
        <Microscope className="w-7 h-7 text-accent-light" />
        <div>
          <h1 className="text-white font-bold text-lg leading-tight">
            MI-Workbench
          </h1>
          <p className="text-indigo-300 text-xs">Mechanistic Interpretability</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-accent text-white"
                  : "text-indigo-200 hover:bg-sidebar-hover hover:text-white"
              }`
            }
          >
            <Icon className="w-5 h-5 flex-shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/10">
        <p className="text-indigo-400 text-xs">v0.1.0</p>
      </div>
    </aside>
  );
}
