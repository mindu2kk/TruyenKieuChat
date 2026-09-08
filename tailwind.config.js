/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./chat_UI/templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f5f7ff",
          100: "#eef2ff",
          600: "#4f46e5",
          700: "#4338ca",
        },
        primary: "#5eead4",
        background: "#18181b",
        card: "#27272a",
        foreground: "#f4f4f5",
        input: "#3f3f46",
      },
    },
  },
  plugins: [],
};
