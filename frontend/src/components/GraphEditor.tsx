import { useCallback, useMemo, useState } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer";
import { Download, Plus } from "lucide-react";
import type { LoopDefinition, LoopNode, LoopEdge } from "../api/client";

// ── Custom Node ──────────────────────────────────────────────────────

const roleColors: Record<string, { bg: string; border: string; text: string }> = {
  executor: { bg: "bg-blue-50", border: "border-blue-400", text: "text-blue-700" },
  reviewer: { bg: "bg-amber-50", border: "border-amber-400", text: "text-amber-700" },
  adversarial_reviewer: { bg: "bg-red-50", border: "border-red-400", text: "text-red-700" },
  idea_generator: { bg: "bg-purple-50", border: "border-purple-400", text: "text-purple-700" },
  knowledge_extractor: { bg: "bg-green-50", border: "border-green-400", text: "text-green-700" },
};

const defaultColors = { bg: "bg-gray-50", border: "border-gray-400", text: "text-gray-700" };

interface NodeData {
  label: string;
  role: string;
  promptRef: string;
}

function LoopNodeComponent({ data }: { data: NodeData }) {
  const colors = roleColors[data.role] ?? defaultColors;
  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 ${colors.border} ${colors.bg} min-w-[160px] shadow-sm`}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-400 !w-2.5 !h-2.5" />
      <div className={`text-sm font-semibold ${colors.text}`}>{data.label}</div>
      <div className="text-xs text-gray-500 mt-0.5">{data.role}</div>
      {data.promptRef && (
        <div className="text-xs text-gray-400 mt-1 truncate">{data.promptRef}</div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-gray-400 !w-2.5 !h-2.5" />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  loopNode: LoopNodeComponent,
};

// ── Graph Editor ─────────────────────────────────────────────────────

interface Props {
  loop: LoopDefinition;
  onChange?: (loop: LoopDefinition) => void;
}

function loopToFlow(loop: LoopDefinition): {
  nodes: Node<NodeData>[];
  edges: Edge[];
} {
  const nodes: Node<NodeData>[] = loop.nodes.map((n, i) => ({
    id: n.id,
    type: "loopNode",
    position: { x: 250 * (i % 3), y: 150 * Math.floor(i / 3) },
    data: { label: n.id, role: n.role, promptRef: n.prompt_ref },
  }));
  const edges: Edge[] = loop.edges.map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    label: e.condition !== "always" ? e.condition : undefined,
    animated: e.condition === "always",
    style: { stroke: "#94a3b8" },
    labelStyle: { fontSize: 11, fill: "#64748b" },
  }));
  return { nodes, edges };
}

function flowToLoop(
  nodes: Node<NodeData>[],
  edges: Edge[],
  base: LoopDefinition,
): LoopDefinition {
  const loopNodes: LoopNode[] = nodes.map((n) => ({
    id: n.id,
    role: n.data.role,
    prompt_ref: n.data.promptRef,
    config: {},
  }));
  const loopEdges: LoopEdge[] = edges.map((e) => ({
    source: e.source,
    target: e.target,
    condition: (e.label as string) || "always",
  }));
  return { ...base, nodes: loopNodes, edges: loopEdges };
}

export default function GraphEditor({ loop, onChange }: Props) {
  const initial = useMemo(() => loopToFlow(loop), [loop]);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [nextId, setNextId] = useState(loop.nodes.length + 1);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, animated: true, style: { stroke: "#94a3b8" } }, eds));
    },
    [setEdges],
  );

  const handleAddNode = useCallback(
    (role: string) => {
      const id = `node_${nextId}`;
      setNextId((n) => n + 1);
      const newNode: Node<NodeData> = {
        id,
        type: "loopNode",
        position: { x: Math.random() * 400, y: Math.random() * 300 },
        data: { label: id, role, promptRef: "" },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [nextId, setNodes],
  );

  const handleExport = useCallback(() => {
    const def = flowToLoop(nodes, edges, loop);
    onChange?.(def);
  }, [nodes, edges, loop, onChange]);

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm text-gray-500 mr-2">Add node:</span>
        {Object.keys(roleColors).map((role) => (
          <button
            key={role}
            onClick={() => handleAddNode(role)}
            className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium border border-surface-border rounded-md hover:bg-gray-50 transition-colors"
          >
            <Plus className="w-3 h-3" />
            {role.replace(/_/g, " ")}
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={handleExport}
          className="flex items-center gap-2 px-3 py-1.5 bg-accent text-white text-xs font-medium rounded-md hover:bg-accent-dark transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          Export
        </button>
      </div>

      {/* Flow canvas */}
      <div className="h-[500px] border border-surface-border rounded-lg overflow-hidden bg-white">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          snapToGrid
          snapGrid={[20, 20]}
        >
          <Background color="#e2e8f0" gap={20} />
          <Controls className="!bg-white !border-surface-border !shadow-md" />
          <MiniMap
            nodeColor={(n) => {
              const role = (n.data as NodeData)?.role;
              const map: Record<string, string> = {
                executor: "#93c5fd",
                reviewer: "#fcd34d",
                adversarial_reviewer: "#fca5a5",
                idea_generator: "#c4b5fd",
                knowledge_extractor: "#6ee7b7",
              };
              return map[role] ?? "#d1d5db";
            }}
            className="!bg-gray-50 !border-surface-border"
          />
        </ReactFlow>
      </div>
    </div>
  );
}
