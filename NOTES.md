# README

## Prerequesites:

```
/docs
    /docs.en.json
    /docs.zh-cn.json
    /docs.zh-tw.json
/resource
    /all-MiniLM-L6-v2
        /model.onnx
        /tokenizer_config.json
        /tokenizer.json
    /text2vec-base-chinese
        /model.onnx
        /tokenizer_config.json
        /tokenizer.json
```

## Build Index

```
cd src
tsc build.ts
node build.js
```

## Test

### Test Retrieval

```
cd src
tsc testRetrieve.ts
node testRetrieve.js
```

### Test Lang

```
cd src
tsc testLang.ts
node testLang.js
```

## Build

```
npm install
npm run compile
npx vsce package
```