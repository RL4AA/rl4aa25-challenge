export default {
  base: process.env.NODE_ENV === 'production' ? '/asdf/' : '/',
  publicDir: 'public', // Explicitly set to the default value
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
          return 'vendor'; // Fallback for other vendor libs
        },
      },
    },
    chunkSizeWarningLimit: 1500, // 1.5 MB
    minify: 'esbuild', // Faster minification
  },
};
