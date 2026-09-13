"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Scenarios" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/sops", label: "SOP lookup" },
];

export function Tabs() {
  const pathname = usePathname();
  return (
    <nav className="tabs" aria-label="Sections">
      {LINKS.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          aria-current={pathname === link.href ? "page" : undefined}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
