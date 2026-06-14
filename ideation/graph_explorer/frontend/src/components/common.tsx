import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

export function cx(...classes: Array<string | false | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function formatRunTime(seconds: number) {
  if (!seconds) return "unknown";
  return new Date(seconds * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function IconButton({
  icon,
  label,
  onClick,
  disabled,
  tone = "default",
}: {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "default" | "primary" | "danger";
}) {
  return (
    <button
      className={cx("btn", tone === "primary" && "btn-primary", tone === "danger" && "btn-danger")}
      disabled={disabled}
      onClick={onClick}
      type="button"
      title={label}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export function Drawer({
  title,
  note,
  icon,
  children,
  defaultOpen = false,
}: {
  title: string;
  note?: string;
  icon?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <Collapsible.Root defaultOpen={defaultOpen} className="drawer">
      <Collapsible.Trigger className="drawer-trigger">
        <span className="drawer-title">
          <ChevronRight className="drawer-chevron" size={14} />
          {icon}
          {title}
        </span>
        {note ? <span className="drawer-note">{note}</span> : null}
      </Collapsible.Trigger>
      <Collapsible.Content className="drawer-content">{children}</Collapsible.Content>
    </Collapsible.Root>
  );
}

export function SidebarHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="sidebar-header">
      <h2>{title}</h2>
      <span>{subtitle}</span>
    </div>
  );
}
