import * as path from 'path';
import { STRetriever } from './src/transformer';

const RES_ROOT = path.join(".", "resource");
const DOCS_EN = path.join(".", "docs", "docs.en.json");
const DOCS_CN = path.join(".", "docs", "docs.zh-cn.json");
const DOCS_TW = path.join(".", "docs", "docs.zh-tw.json");
const INDEX_EN = path.join(RES_ROOT, "index.en.json");
const INDEX_CN = path.join(RES_ROOT, "index.zh-cn.json");
const INDEX_TW = path.join(RES_ROOT, "index.zh-tw.json");
const MODEL_ROOT_EN = path.join(RES_ROOT, "all-MiniLM-L6-v2");
const MODEL_ROOT_CN = path.join(RES_ROOT, "text2vec-base-chinese");
const MODEL_ROOT_TW = path.join(RES_ROOT, "text2vec-base-chinese");

async function runBuild() {
    console.log("Building index ...");
    try {
        await STRetriever.createAndBuild(
            path.join(MODEL_ROOT_EN, "model.onnx"),
            path.join(MODEL_ROOT_EN, "tokenizer.json"),
            path.join(MODEL_ROOT_EN, "tokenizer_config.json"),
            DOCS_EN
        ).then(r => {
            r.saveIndex(INDEX_EN);
            console.log(`Created ${INDEX_EN}`);
        });

        await STRetriever.createAndBuild(
            path.join(MODEL_ROOT_CN, "model.onnx"),
            path.join(MODEL_ROOT_CN, "tokenizer.json"),
            path.join(MODEL_ROOT_CN, "tokenizer_config.json"),
            DOCS_CN
        ).then(r => {
            r.saveIndex(INDEX_CN);
            console.log(`Created ${INDEX_CN}`);
        });

        await STRetriever.createAndBuild(
            path.join(MODEL_ROOT_TW, "model.onnx"),
            path.join(MODEL_ROOT_TW, "tokenizer.json"),
            path.join(MODEL_ROOT_TW, "tokenizer_config.json"),
            DOCS_TW
        ).then(r => {
            r.saveIndex(INDEX_TW);
            console.log(`Created ${INDEX_TW}`);
        });

    } catch (error) {
        console.error("Test execution failed:", error);
    }
}

runBuild().catch(console.error);
