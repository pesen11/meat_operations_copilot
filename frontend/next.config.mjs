/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // Where the FastAPI backend lives. Set NEXT_PUBLIC_API_URL in Vercel.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000",
  },
};
export default nextConfig;
