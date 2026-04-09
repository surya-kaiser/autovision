/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0f1117',
          800: '#1a1d27',
          700: '#242837',
          600: '#2e3346',
        },
        accent: {
          500: '#6366f1',
          400: '#818cf8',
        },
      },
    },
  },
  plugins: [],
}
