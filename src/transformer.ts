import { Tokenizer } from "@huggingface/tokenizers";
import { FuncDocument, FuncDocumentMap } from "./docs";

import * as ort from "onnxruntime-node";
import * as fs from 'fs';

type IndexedDoc = {
    id: string;
    doc: FuncDocument;
    embeddings: Float32Array[];
};

type IndexDocJson = {
    id: string;
    doc: FuncDocument;
    embeddings: number[][];
}

export class STRetriever {
    private static readonly MAX_LEN = 512;
    private static readonly CHUNK_OVERLAP = 50;
    private static readonly PAD_ID = 0;
    private static readonly HIDDEN_SIZE = 384; // Standard for MiniLM

    private session: ort.InferenceSession | null = null;
    private index: IndexedDoc[] = [];
    private tokenizer: Tokenizer | null = null;

    private constructor() { }

    public static async create(
        onnxModelFile: string,
        tokenizerFile: string,
        tokenizerConfigFile: string
    ): Promise<STRetriever> {
        const instance = new STRetriever();
        instance.session = await ort.InferenceSession.create(onnxModelFile);

        const tokenizer = fs.readFileSync(tokenizerFile, "utf-8");
        const tokenizerConfig = fs.readFileSync(tokenizerConfigFile, "utf-8");
        instance.tokenizer = new Tokenizer(JSON.parse(tokenizer), JSON.parse(tokenizerConfig));

        return instance;
    }

    public static async createAndBuild(
        onnxModelFile: string, tokenizerFile: string,
        tokenizerConfigFile: string, docFile: string
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

    public async buildIndex(docFile: string) {
        const docs: FuncDocumentMap = JSON.parse(fs.readFileSync(docFile, 'utf8'));

        for (const id in docs) {
            const doc = docs[id];
            const text = doc.comments.join(" ");
            const encoded = this.tokenizer!.encode(text);

            const chunkEmbeddings: Float32Array[] = [];
            const step = STRetriever.MAX_LEN - STRetriever.CHUNK_OVERLAP;

            // Sliding window over token IDs
            for (let i = 0; i < encoded.ids.length; i += step) {
                const chunkIds = encoded.ids.slice(i, i + STRetriever.MAX_LEN);
                const chunkMask = encoded.attention_mask.slice(i, i + STRetriever.MAX_LEN);

                const embedding = await this.embedRaw(chunkIds, chunkMask);
                chunkEmbeddings.push(embedding);

                if (i + STRetriever.MAX_LEN >= encoded.ids.length) break;
            }

            this.index.push({ id, doc, embeddings: chunkEmbeddings });
        }
    }

    private async embedRaw(ids: number[], mask: number[]): Promise<Float32Array> {
        // Manual Pad
        let inputIds = [...ids];
        let attentionMask = [...mask];
        const padLength = STRetriever.MAX_LEN - inputIds.length;

        if (padLength > 0) {
            inputIds = inputIds.concat(new Array(padLength).fill(STRetriever.PAD_ID));
            attentionMask = attentionMask.concat(new Array(padLength).fill(0));
        }

        const feeds = {
            input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, STRetriever.MAX_LEN]),
            attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, STRetriever.MAX_LEN]),
            token_type_ids: new ort.Tensor("int64", new BigInt64Array(STRetriever.MAX_LEN).fill(0n), [1, STRetriever.MAX_LEN])
        };

        const results = await this.session!.run(feeds);
        const pooled = this.meanPooling(
            results.last_hidden_state.data as Float32Array,
            BigInt64Array.from(attentionMask.map(BigInt)),
            STRetriever.HIDDEN_SIZE
        );
        return this.l2Normalize(pooled);
    }

    private async embed(text: string): Promise<Float32Array> {
        const encoded = this.tokenizer!.encode(text);
        return this.embedRaw(
            encoded.ids.slice(0, STRetriever.MAX_LEN),
            encoded.attention_mask.slice(0, STRetriever.MAX_LEN)
        );
    }

    public async retrieve(text: string, topK: number): Promise<FuncDocument[]> {
        const queryEmbedding = await this.embed(text);

        const scored = this.index.map(entry => {
            const maxScore = Math.max(...entry.embeddings.map(chunkEmb =>
                this.cosineSimilarity(queryEmbedding, chunkEmb)
            ));
            return { doc: entry.doc, score: maxScore };
        });

        return scored
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .map(s => s.doc);
    }

    public async retrieveForTest(text: string, topK: number, expectedResult: string): Promise<number> {
        if (this.index.length === 0) {
            throw new Error("No documents indexed");
        }

        const queryEmbedding = await this.embed(text);

        const scored = this.index.map(entry => {
            const maxScore = Math.max(...entry.embeddings.map(chunkEmb =>
                this.cosineSimilarity(queryEmbedding, chunkEmb)
            ));

            return {
                id: entry.id,
                score: maxScore
            };
        });

        scored.sort((a, b) => b.score - a.score);

        const result = scored.slice(0, topK);
        const scores = [1, 0.5, 0.25, 0.125, 0.0625];
        const matchIndex = result.findIndex(r => r.id === expectedResult);

        return matchIndex !== -1 ? (scores[matchIndex] ?? 0) : 0;
    }

    public async saveIndex(indexFile: string) {
        const indexJson: IndexDocJson[] = this.index.map(doc => ({
            id: doc.id,
            doc: doc.doc,
            embeddings: doc.embeddings.map(e => Array.from(e))
        }));
        fs.writeFileSync(indexFile, JSON.stringify(indexJson), "utf-8");
    }

    public async loadIndex(indexFile: string) {
        const raw = fs.readFileSync(indexFile, "utf-8");
        const data = JSON.parse(raw) as IndexDocJson[];
        this.index = data.map(item => ({
            id: item.id,
            doc: item.doc,
            embeddings: item.embeddings.map(e => new Float32Array(e))
        }));
    }

    private meanPooling(embeddings: Float32Array, mask: BigInt64Array, hiddenSize: number): Float32Array {
        const pooled = new Float32Array(hiddenSize);
        let count = 0;
        for (let i = 0; i < mask.length; i++) {
            if (mask[i] === 1n) {
                count++;
                for (let j = 0; j < hiddenSize; j++) {
                    pooled[j] += embeddings[i * hiddenSize + j];
                }
            }
        }
        return pooled.map(v => v / count);
    }

    private l2Normalize(vec: Float32Array): Float32Array {
        const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
        return vec.map(v => v / norm);
    }

    private cosineSimilarity(a: Float32Array, b: Float32Array): number {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }
}