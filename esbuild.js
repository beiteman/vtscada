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
    external: ['vscode', 'onnxruntime-node', '*.node', "@node-rs/jieba-*"],
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
    const foldersToCopy = [
        'onnxruntime-node',
        'onnxruntime-common',
        '@huggingface/tokenizers',
        '@node-rs',
        'stemmer'
    ];
    for (const pkg of foldersToCopy) {
        await copyFolderToDist(pkg);
    }
    await ctx.dispose();
  }
}

async function copyFolderToDist(pkgName) {
    const src = path.join(__dirname, 'node_modules', pkgName);
    const dest = path.join(__dirname, 'dist', 'node_modules', pkgName);
    if (require('fs').existsSync(src)) {
        copySync(src, dest, { overwrite: true });
        console.log(`[copy] ${pkgName} copied to dist`);
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
