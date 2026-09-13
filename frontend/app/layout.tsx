import type { Metadata } from "next";
import "./globals.css";
import { Tabs } from "@/components/Tabs";
import { STRIP_EXTENSION_ATTRS } from "./strip-extension-attrs";

export const metadata: Metadata = {
  title: "Meat-Cutting Operations Copilot",
  description:
    "Simulate production decisions for a butcher shop: inventory, labor, waste and margin, computed deterministically.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {process.env.NODE_ENV !== "production" ? (
          <script dangerouslySetInnerHTML={{ __html: STRIP_EXTENSION_ATTRS }} />
        ) : null}
      </head>
      <body suppressHydrationWarning>
        <div className="shell">
          <header className="masthead">
            <h1>Meat-Cutting Operations Copilot</h1>
            <Tabs />
          </header>
          <p className="tagline">
            Every figure below is computed by deterministic, unit-tested Python.
            The language model routes the question and writes the summary; it
            never produces a number.
          </p>
          {children}
        </div>
      </body>
    </html>
  );
}
