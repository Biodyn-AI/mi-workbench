/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sidebar: {
          DEFAULT: "#1e1b4b",
          light: "#312e81",
          hover: "#3730a3",
        },
        accent: {
          DEFAULT: "#3b82f6",
          dark: "#2563eb",
          light: "#60a5fa",
        },
        surface: {
          DEFAULT: "#f8fafc",
          card: "#ffffff",
          border: "#e2e8f0",
        },
      },
    },
  },
  plugins: [],
};
