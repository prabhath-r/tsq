import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "TSQ — The Second Question";
const description =
  "A precise, explainable adaptive-learning workspace for machine learning.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const forwardedHost = requestHeaders.get("x-forwarded-host")?.split(",")[0]?.trim();
  const requestHost = forwardedHost ?? requestHeaders.get("host") ?? "localhost:3000";
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto")?.split(",")[0]?.trim();
  const requestProtocol = forwardedProtocol ?? (requestHost.startsWith("localhost") ? "http" : "https");
  let origin: URL;

  try {
    origin = new URL(`${requestProtocol}://${requestHost}`);
  } catch {
    origin = new URL("http://localhost:3000");
  }

  return {
    metadataBase: origin,
    title,
    description,
    applicationName: "TSQ",
    icons: {
      icon: [{ url: "/icon.png", type: "image/png", sizes: "128x128" }],
      apple: [{ url: "/icon.png", type: "image/png", sizes: "128x128" }],
    },
    openGraph: {
      type: "website",
      url: origin.toString(),
      siteName: "TSQ",
      title,
      description,
    },
    twitter: {
      card: "summary",
      title,
      description,
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("tsq-theme");var dark=t==="dark"||(t!=="light"&&matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.dataset.theme=dark?"dark":"light";document.documentElement.dataset.reducedMotion=localStorage.getItem("tsq-reduce-motion")==="true"?"true":"false"}catch(e){}})();`,
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
