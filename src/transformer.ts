// npm install onnxruntime-node @huggingface/tokenizers

import { Tokenizer } from "@huggingface/tokenizers";
import { FuncDocument, FuncDocumentMap } from "./docs";

import * as ort from "onnxruntime-node";
import * as fs from 'fs';
import { integer } from "vscode-languageclient";

type IndexedDoc = {
    id: string;
    doc: FuncDocument;
    embedding: Float32Array;
};

type IndexDocJson = {
    id: string;
    doc: FuncDocument;
    embedding: number[];
}

export class STRetriever {
    private static readonly MAX_LEN = 512;
    private static readonly PAD_ID = 0; // correct for BERT/MiniLM

    private session: ort.InferenceSession | null = null;
    private index: IndexedDoc[] = [];
    private tokenizer: Tokenizer | null = null;

    // use creator instead
    private constructor() {
    }

    public static async create(
        onnxModelFile: string,
        tokenizerFile: string,
        tokenizerConfigFile: string
    ): Promise<STRetriever> {
        if (!fs.existsSync(onnxModelFile)) {
            throw new Error(`Model file not found: ${onnxModelFile}`);
        }
        if (!fs.existsSync(tokenizerFile)) {
            throw new Error(`Tokenizer file not found: ${tokenizerFile}`);
        }
        if (!fs.existsSync(tokenizerConfigFile)) {
            throw new Error(`Tokenizer config file not found: ${tokenizerConfigFile}`);
        }

        const instance = new STRetriever();

        // load model
        instance.session = await ort.InferenceSession.create(onnxModelFile);

        // load tokenizer 
        const tokenizer = fs.readFileSync(tokenizerFile, "utf-8");
        const tokenizerConfig = fs.readFileSync(tokenizerConfigFile, "utf-8");
        const tokenizerJson = JSON.parse(tokenizer);
        const tokenizerConfigJson = JSON.parse(tokenizerConfig);
        instance.tokenizer = new Tokenizer(tokenizerJson, tokenizerConfigJson);

        return instance;
    }

    public static async createAndBuild(
        onnxModelFile: string,
        tokenizerFile: string,
        tokenizerConfigFile: string,
        docFile: string
    ): Promise<STRetriever> {
        const instance = await this.create(onnxModelFile, tokenizerFile, tokenizerConfigFile);
        await instance.buildIndex(docFile);
        return instance;
    }

    public static async createAndLoad(
        onnxModelFile: string,
        tokenizerFile: string,
        tokenizerConfigFile: string,
        indexFile: string
    ): Promise<STRetriever> {
        const instance = await this.create(onnxModelFile, tokenizerFile, tokenizerConfigFile);
        await instance.loadIndex(indexFile);
        return instance;
    }

    public async loadIndex(indexFile: string) {
        if (!fs.existsSync(indexFile)) {
            throw new Error(`Index file not found: ${indexFile}`);
        }

        const raw = fs.readFileSync(indexFile, "utf-8");
        const index = JSON.parse(raw) as IndexDocJson[];
        this.index = index
            .filter(doc => doc.doc.snippets.length > 0) // exclude items without snippets
            .map(doc => ({
                id: doc.id,
                doc: doc.doc,
                embedding: new Float32Array(doc.embedding)
            }));
    }

    public async saveIndex(indexFile: string) {
        if (this.index.length === 0) {
            throw new Error("No documents to save");
        }
        const indexJson = this.index.map(doc => ({
            id: doc.id,
            doc: doc.doc,
            embedding: Array.from(doc.embedding)
        }));
        fs.writeFileSync(indexFile, JSON.stringify(indexJson), "utf-8");
    }

    public async buildIndex(docFile: string) {
        // load docs
        const docs: FuncDocumentMap = JSON.parse(fs.readFileSync(docFile, 'utf8'));

        // vectorize docs
        for (const id in docs) {
            if (Object.prototype.hasOwnProperty.call(docs, id)) {
                const doc = docs[id];
                const text = doc.comments.join(" ");
                const embedding = await this.embed(text);
                this.index.push({ id: id, doc: doc, embedding });
            }
        }
    }

    private tokenize(text: string) {
        if (this.tokenizer == null) {
            throw new Error("Initialization is not yet done");
        }
        const encoded = this.tokenizer.encode(text);
        let inputIds = encoded.ids;
        let attentionMask = encoded.attention_mask;
        const length = encoded.ids.length;

        // (MANUAL) TRUNCATE
        if (inputIds.length > STRetriever.MAX_LEN) {
            inputIds = inputIds.slice(0, STRetriever.MAX_LEN);
            attentionMask = attentionMask.slice(0, STRetriever.MAX_LEN);
        }

        // (MANUAL) PAD
        const padLength = STRetriever.MAX_LEN - inputIds.length;

        if (padLength > 0) {
            inputIds = inputIds.concat(
                new Array(padLength).fill(STRetriever.PAD_ID)
            );
            attentionMask = attentionMask.concat(
                new Array(padLength).fill(0)
            );
        }

        // all zero
        const tokenTypeIds = new Array(STRetriever.MAX_LEN).fill(0);

        return {
            input_ids: BigInt64Array.from(inputIds.map(BigInt)),
            attention_mask: BigInt64Array.from(attentionMask.map(BigInt)),
            token_type_ids: BigInt64Array.from(tokenTypeIds.map(BigInt))
        };
    }

    private meanPooling(
        tokenEmbeddings: Float32Array,
        attentionMask: BigInt64Array,
        hiddenSize: number
    ): Float32Array {
        const pooled = new Float32Array(hiddenSize);
        let validTokens = 0;
        for (let i = 0; i < attentionMask.length; i++) {
            if (attentionMask[i] === 1n) {
                validTokens++;
                for (let j = 0; j < hiddenSize; j++) {
                    pooled[j] += tokenEmbeddings[i * hiddenSize + j];
                }
            }
        }
        for (let j = 0; j < hiddenSize; j++) {
            pooled[j] /= validTokens;
        }
        return pooled;
    }

    private l2Normalize(vec: Float32Array) {
        let norm = 0;
        for (const v of vec) norm += v * v;
        norm = Math.sqrt(norm);

        for (let i = 0; i < vec.length; i++) {
            vec[i] /= norm;
        }
        return vec;
    }

    private cosineSimilarity(a: Float32Array, b: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private async embed(text: string): Promise<Float32Array> {
        if (this.session == null) {
            throw new Error("Initialization is not yet done");
        }

        const { input_ids, attention_mask, token_type_ids } = this.tokenize(text);

        const feeds = {
            input_ids: new ort.Tensor("int64", input_ids, [1, input_ids.length]),
            attention_mask: new ort.Tensor("int64", attention_mask, [1, attention_mask.length]),
            token_type_ids: new ort.Tensor("int64", token_type_ids, [1, attention_mask.length])
        };

        const results = await this.session.run(feeds);

        const lastHiddenState = results.last_hidden_state.data as Float32Array;
        const hiddenSize = 384;

        const pooled = this.meanPooling(
            lastHiddenState,
            attention_mask,
            hiddenSize
        );

        return this.l2Normalize(pooled);
    }

    public async retrieve(text: string, topK: integer): Promise<FuncDocument[]> {
        if (this.index.length === 0) {
            throw new Error("No documents indexed");
        }
        const queryEmbedding = await this.embed(text);
        const scored = this.index.map(doc => ({
            id: doc.id,
            doc: doc.doc,
            score: this.cosineSimilarity(queryEmbedding, doc.embedding)
        }));
        scored.sort((a, b) => b.score - a.score);
        const result = scored.slice(0, topK);
        const resultDocs = result.map(item => item.doc);
        console.log(`Q: ${text}`);
        result.forEach(r => {
            console.log(r.score, r.id, r.doc.snippets[0]);
        })
        return resultDocs;
    }

    public async retrieveForTest(text: string, topK: integer, expectedResult: string): Promise<number> {
        if (this.index.length === 0) {
            throw new Error("No documents indexed");
        }
        const queryEmbedding = await this.embed(text);
        const scored = this.index.map(doc => ({
            id: doc.id,
            doc: doc.doc,
            score: this.cosineSimilarity(queryEmbedding, doc.embedding)
        }));
        scored.sort((a, b) => b.score - a.score);
        const result = scored.slice(0, topK);
        const scores = [1, 0.5, 0.25, 0.125, 0.0625];
        const matchIndex = result.findIndex(r => r.id === expectedResult);
        const score = scores[matchIndex] ?? 0;
        return score;
    }

}