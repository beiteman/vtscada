"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const ort = __importStar(require("onnxruntime-node"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const natural_1 = require("natural");
// Load model info (vocabulary, docs)

const modelPath = 'model.zh-cn.onnx';
const modelInfoPath = path.join(__dirname, 'info.zh-cn.json');

const modelInfo = JSON.parse(fs.readFileSync(modelInfoPath, 'utf8'));
const keys = modelInfo.keys;
const vocabulary = modelInfo.vocabulary;
const docs = modelInfo;
const vocabSize = Object.keys(vocabulary).length;
// --- Preprocessing
function analyzeQuery(query, useStemming = true) {
    let qClean = query.replace(/[^\w\s]/g, " ").toLowerCase();
    let tokens = qClean.split(/\s+/).filter(t => t.length > 0 && !natural_1.stopwords.includes(t));
    if (useStemming) {
        tokens = tokens.map(t => natural_1.PorterStemmer.stem(t));
    }
    return tokens;
}
/**
 * Creates a dense TFIDF vector from the query tokens.
 * NOTE: this assumes a CountVectorizer/TF-only model
 */
function vectorizeQuery(queryTokens) {
    const vector = new Array(vocabSize).fill(0);
    for (const token of queryTokens) {
        const index = vocabulary[token];
        if (index !== undefined) {
            // Simple Term Frequency (TF)
            vector[index] += 1;
        }
    }
    // Create ONNX Tensor
    const queryVector = new ort.Tensor('float32', Float32Array.from(vector), [1, vocabSize]);
    return queryVector;
}
// --- Inference ---
async function infer(query, topK) {
    const session = await ort.InferenceSession.create(modelPath);
    // 1. Preprocess Query & Vectorize
    const tokens = analyzeQuery(query);
    const queryVector = vectorizeQuery(tokens);
    // 2. Run ONNX Inference
    const feeds = { 'query_vector': queryVector };
    const results = await session.run(feeds);
    // The output is the dot product score (similarity_scores)
    const scores = results['similarity_scores'].data;
    // 3. Post-process (Sort Results)
    const rankedResults = Array.from(scores).map((score, index) => ({
        index,
        score: score,
        key: keys[index]
    }));
    rankedResults.sort((a, b) => b.score - a.score);
    // 4. Print & Return Top K
    const topResults = rankedResults.slice(0, topK);
    console.log(`\nQuery: '${query}'`);
    console.log('-'.repeat(60));
    topResults.forEach((result, rank) => {
        console.log(result.key, result.score);
    });
    console.log('-'.repeat(60));
    return topResults.map(r => docs[r.index]);
}
// --- Main Execution ---
// infer("draw", 5).catch(console.error); // en
// infer("確認警報", 5).catch(console.error); // tw
infer("以文本字符串形式返回任何命令行参数", 5).catch(console.error); // cn

