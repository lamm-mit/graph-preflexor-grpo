import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, Info } from "lucide-react";
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
  description,
}: {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "default" | "primary" | "danger";
  description?: string;
}) {
  const title = description ? `${label}: ${description}` : label;
  return (
    <button
      className={cx("btn", tone === "primary" && "btn-primary", tone === "danger" && "btn-danger")}
      disabled={disabled}
      onClick={onClick}
      type="button"
      title={title}
      aria-label={title}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export function HelpTip({ text }: { text: string }) {
  return (
    <span className="help-tip" title={text} aria-label={text}>
      <Info size={12} />
    </span>
  );
}

export function Drawer({
  title,
  note,
  icon,
  children,
  defaultOpen = false,
  description,
}: {
  title: string;
  note?: string;
  icon?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  description?: string;
}) {
  return (
    <Collapsible.Root defaultOpen={defaultOpen} className="drawer">
      <Collapsible.Trigger className="drawer-trigger" title={description || title}>
        <span className="drawer-title">
          <ChevronRight className="drawer-chevron" size={14} />
          {icon}
          {title}
          {description ? <HelpTip text={description} /> : null}
        </span>
        {note ? <span className="drawer-note">{note}</span> : null}
      </Collapsible.Trigger>
      <Collapsible.Content className="drawer-content">
        {description ? <div className="drawer-help">{description}</div> : null}
        {children}
      </Collapsible.Content>
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
