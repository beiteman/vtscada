// esbuild.js
const esbuild = require('esbuild');
const { copySync, ensureDirSync } = require('fs-extra');
const path = require('path');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

async function main() {
  const ctx = await esbuild.context({
    entryPoints: ['src/extension.ts'],
    bundle: true,
    format: 'cjs',
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: 'node',
    outfile: 'dist/extension.js',
    // keep onnxruntime-node external so the native addon isn't bundled
    external: ['vscode', 'onnxruntime-node'],
    logLevel: 'warning',
    plugins: [
      /* existing plugin(s) */
      esbuildProblemMatcherPlugin
    ]
  });

  if (watch) {
    await ctx.watch();
  } else {
    // run a single build
    await ctx.rebuild();
    // After successful build, copy the native artifacts we need
    await copyOnnxruntimeNodeFiles();
    await ctx.dispose();
  }
}

/**
 * Copy the runtime files from node_modules/onnxruntime-node into dist so runtime require() finds
 * the native .node files in the same relative path (e.g. ../bin/napi-v6/darwin/x64/onnxruntime_binding.node).
 */
async function copyOnnxruntimeNodeFiles() {
  try {
    const pkgRoot = path.join(__dirname, 'node_modules', 'onnxruntime-node');
    const destRoot = path.join(__dirname, 'dist', 'node_modules', 'onnxruntime-node');

    // Ensure destination folder exists
    ensureDirSync(destRoot);

    // 1) Copy package.json (some modules inspect package metadata)
    const srcPkg = path.join(pkgRoot, 'package.json');
    try {
      copySync(srcPkg, path.join(destRoot, 'package.json'), { overwrite: true });
      console.log('[copy] onnxruntime-node/package.json copied');
    } catch (e) {
      // package.json may not exist in some installs—but normally it does
      console.warn('[copy] could not copy package.json for onnxruntime-node:', e.message);
    }

    // 2) Copy the bin directory (contains napi builds and .node files)
    const srcBin = path.join(pkgRoot, 'bin');
    try {
      copySync(srcBin, path.join(destRoot, 'bin'), { overwrite: true, recursive: true });
      console.log('[copy] onnxruntime-node/bin copied');
    } catch (e) {
      console.warn('[copy] could not copy onnxruntime-node/bin:', e.message);
    }

    // Optional: copy any other files you might need (README, dist, build, etc.)
    // copySync(path.join(pkgRoot, 'dist'), path.join(destRoot, 'dist'), { overwrite: true, recursive: true });

    console.log('[copy] onnxruntime-node runtime files copied to dist');
  } catch (err) {
    console.error('[copy] failed to copy onnxruntime-node files:', err);
    // keep the error non-fatal so you can still inspect build output; decide to exit if you prefer
  }
}

/**
 * Problem-matcher plugin you had
 * @type {import('esbuild').Plugin}
 */
const esbuildProblemMatcherPlugin = {
  name: 'esbuild-problem-matcher',
  setup(build) {
    build.onStart(() => {
      console.log('[watch] build started');
    });
    build.onEnd(result => {
      result.errors.forEach(({ text, location }) => {
        console.error(`✘ [ERROR] ${text}`);
        if (location == null) return;
        console.error(`    ${location.file}:${location.line}:${location.column}:`);
      });
      console.log('[watch] build finished');
    });
  }
};

main().catch(e => {
  console.error(e);
  process.exit(1);
});
