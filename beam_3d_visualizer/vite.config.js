import { defineConfig } from 'vite';
import { copy } from 'vite-plugin-copy';

export default defineConfig(({ mode }) => {
  // Load environment variables based on mode
  const base = process.env.VITE_BASE_DIR || '/';

  return {
    define: {
      'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    },
    envPrefix: 'VITE_', // Stick to Vite convention
    base, // Use the dynamically determined base
    publicDir: 'public',
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules/three')) {
              return 'three';
            }
            if (id.includes('node_modules/plotly.js-dist')) {
              return 'plotly';
            }
            if (id.includes('node_modules/three/examples/jsm')) {
              return 'three-extras';
            }
            return 'vendor';
          },
        },
      },
      chunkSizeWarningLimit: 1500,  // 1.5 MB
      minify: 'esbuild',  // Faster minification
    },
    plugins: [
      copy({
        targets: [
          { src: 'serve.json', dest: 'dist' }
        ]
      })
    ]
  };
});
