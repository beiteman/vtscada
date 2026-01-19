import * as path from 'path';
import { STRetriever } from './transformer';

const resPath = path.join("C:/Users/andre/Documents/vtscada-function-retrieval", "resources");

// // shibing624/text2vec-base-chinese
// const stPath = path.join(resPath, "shibing624", "text2vec-base-chinese");
// const indexPath = path.join(resPath, "text2vec-base-chinese.zh-cn.json");
// const stModelPath = path.join(stPath, "onnx", "model.onnx");
// const stTkrPath = path.join(stPath, "onnx", "tokenizer.json");
// const stTkrConfigPath = path.join(stPath, "onnx", "tokenizer_config.json");
// const docsPath = path.join(resPath, "docs.zh-cn.json");

// shibing624/text2vec-base-chinese
const stPath = path.join(resPath, "shibing624", "text2vec-base-chinese");
const indexPath = path.join(resPath, "text2vec-base-chinese.zh-tw.json");
const stModelPath = path.join(stPath, "onnx", "model.onnx");
const stTkrPath = path.join(stPath, "onnx", "tokenizer.json");
const stTkrConfigPath = path.join(stPath, "onnx", "tokenizer_config.json");
const docsPath = path.join(resPath, "docs.zh-tw.json");

// // sentence-transformers/all-MiniLM-L6-v2
// const stPath = path.join(resPath, "sentence-transformers", "all-MiniLM-L6-v2");
// const indexPath = path.join(resPath, "all-MiniLM-L6-v2.en.json");
// const stModelPath = path.join(stPath, "onnx", "model.onnx");
// const stTkrPath = path.join(stPath, "tokenizer.json");
// const stTkrConfigPath = path.join(stPath, "tokenizer_config.json");
// const docsPath = path.join(resPath, "docs.en.json");

function sentenceTransformerBuild() {
    STRetriever.createAndBuild(stModelPath, stTkrPath, stTkrConfigPath, docsPath).then(retriever => {
        console.log("Loaded!");
        retriever.saveIndex(indexPath);
        retriever.retrieve("draws a rectangle", 5);
        retriever.retrieve("畫一個橢圓", 10);
    }).catch(error => {
        console.error(error);
        return Promise.reject(error);
    });
}