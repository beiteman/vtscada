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
exports.FuncDocumentRetriever = void 0;
const ort = __importStar(require("onnxruntime-node"));
const fs = __importStar(require("fs"));
const natural_1 = require("natural");
class FuncDocumentRetriever {
    // use creator instead
    constructor() {
        this.session = null;
        this.vocabulary = null;
        this.keys = []; // function name
        this.vocabSize = 0;
        this.docs = {};
    }
    static async create(onnxModelFile, modelInfoFile, docFile) {
        const instance = new FuncDocumentRetriever();
        console.log(`[INFO] Loading model info from: ${modelInfoFile}`);
        const docs = JSON.parse(fs.readFileSync(docFile, 'utf8'));
        const modelInfo = JSON.parse(fs.readFileSync(modelInfoFile, 'utf8'));
        instance.vocabulary = modelInfo.vocabulary;
        instance.keys = modelInfo.keys;
        instance.docs = docs;
        instance.vocabSize = Object.keys(modelInfo.vocabulary).length;
        console.log(`[INFO] Vocabulary size: ${instance.vocabSize}`);
        console.log(`[INFO] Creating ONNX Inference Session for: ${onnxModelFile}`);
        instance.session = await ort.InferenceSession.create(onnxModelFile);
        return instance;
    }
    analyzeQuery(query, useStemming = true) {
        let qClean = query.replace(/[^\w\s]/g, " ").toLowerCase();
        let tokens = qClean.split(/\s+/).filter(t => t.length > 0 && !natural_1.stopwords.includes(t));
        if (useStemming) {
            tokens = tokens.map(t => natural_1.PorterStemmer.stem(t));
        }
        return tokens;
    }
    vectorizeQuery(queryTokens) {
        if (this.vocabulary == null) {
            throw new Error('this.vocabulary is null');
        }
        const vector = new Array(this.vocabSize).fill(0);
        for (const token of queryTokens) {
            const index = this.vocabulary[token];
            if (index !== undefined) {
                vector[index] += 1;
            }
        }
        const queryVector = new ort.Tensor('float32', Float32Array.from(vector), [1, this.vocabSize]);
        return queryVector;
    }
    async retrieve(query, topK = 5) {
        if (this.session == null)
            throw new Error('this.session is null');
        // Preprocess
        const tokens = this.analyzeQuery(query);
        const queryVector = this.vectorizeQuery(tokens);
        // ONNX Inference
        const feeds = { 'query_vector': queryVector };
        const results = await this.session.run(feeds);
        // Process and Rank Results
        const scores = results['similarity_scores'].data;
        let theResults = Array.from(scores).map((score, index) => ({
            index,
            score: score,
            key: this.keys[index]
        }));
        theResults = theResults.filter(item => item.key in this.docs);
        theResults.sort((a, b) => b.score - a.score);
        theResults = theResults.slice(0, topK);
        return theResults.map(r => this.docs[r.key]);
    }
}
exports.FuncDocumentRetriever = FuncDocumentRetriever;
