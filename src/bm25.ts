import { Jieba } from '@node-rs/jieba'
import { dict } from '@node-rs/jieba/dict'
import { FuncDocument, FuncDocumentMap } from "./docs";
import * as fs from 'fs';

// --------------------------------------------------------
// Types
// --------------------------------------------------------
export interface BM25Options {
    k1?: number;
    b?: number;
    tokenizer?: (text: string) => string[];
    enableChinese?: boolean;  // if true → use jieba
}

export interface BM25Document {
    id: string;
    text: string;
}

// --------------------------------------------------------
// BM25 Class
// --------------------------------------------------------
export class BM25Retriever {
    private bm25: BM25 | null = null;
    private docs: FuncDocumentMap = {};

    // use creator instead
    private constructor() {
    }

    public static async create(
        docPath: string,
        isChinese: boolean
    ): Promise<BM25Retriever> {
        const instance = new BM25Retriever();
        instance.docs = JSON.parse(fs.readFileSync(docPath, 'utf8'));
        const docs: {id: string, text: string}[] = [];
        for (const id in instance.docs) {
        	if (Object.prototype.hasOwnProperty.call(instance.docs, id)) {
        		const doc = instance.docs[id];
        		const text = doc.comments.join(" ");
                const snippets = doc.snippets;
                if (snippets.length > 0) {
        		    docs.push({id: id, text: text});
                }
        	}
        }
        instance.bm25 = new BM25(docs, { enableChinese: isChinese });
        return instance;
    }

    public async retrieve(query: string, topK: number = 5): Promise<FuncDocument[]> {
        if (this.bm25 == null) throw new Error('this.session is null');
        const result = this.bm25.search(query, topK);
        console.log(result);
        result.forEach(r => console.log(r));
        return result.map(r => this.docs[r.id]);
    }
}

export class BM25 {
    private docs: BM25Document[];
    private tokenizer: (text: string) => string[];
    private k1: number;
    private b: number;

    private corpusTokens: string[][] = [];
    private docFreq: Map<string, number> = new Map();
    private avgDocLen = 0;
    private N = 0;

    constructor(docs: BM25Document[], options?: BM25Options) {
        this.docs = docs;
        this.k1 = options?.k1 ?? 1.5;
        this.b = options?.b ?? 0.75;

        // Setup tokenizer
        if (options?.tokenizer) {
            this.tokenizer = options.tokenizer;
        } else if (options?.enableChinese) {
            // Chinese tokenizer using jieba
            const jieba = Jieba.withDict(dict)
            this.tokenizer = (text: string) =>
                jieba.cut(text, false).filter(t => t.trim().length > 0);
        } else {
            // Default English tokenizer
            this.tokenizer = this.defaultEnglishTokenizer.bind(this);
        }

        this.build();
    }

    private defaultEnglishTokenizer(text: string): string[] {
        if (!text) return [];
        // Normalize to NFKD then remove non-ascii letters/digits, collapse whitespace
        // Keep things simple and robust.
        const normalized = text
            .normalize("NFKD")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, " ")
            .trim();
        if (normalized.length === 0) return [];
        return normalized.split(/\s+/).filter((t) => t.length > 0);
    }

    // --------------------------------------------------------
    // Build Index
    // --------------------------------------------------------
    private build(): void {
        this.N = this.docs.length;

        let totalLen = 0;

        for (const doc of this.docs) {
            const tokens = this.tokenizer(doc.text);
            this.corpusTokens.push(tokens);

            totalLen += tokens.length;

            const unique = new Set(tokens);
            unique.forEach(token => {
                this.docFreq.set(token, (this.docFreq.get(token) || 0) + 1);
            });
        }

        this.avgDocLen = totalLen / this.N;
    }

    // --------------------------------------------------------
    // Scoring: BM25 formula
    // --------------------------------------------------------
    private scoreOneDoc(queryTokens: string[], docTokens: string[]): number {
        const freq: Record<string, number> = {};
        for (const t of docTokens) {
            freq[t] = (freq[t] || 0) + 1;
        }

        let score = 0;

        for (const q of queryTokens) {
            if (!freq[q]) continue;

            const df = this.docFreq.get(q) || 0;
            const idf = Math.log(1 + (this.N - df + 0.5) / (df + 0.5));

            const fq = freq[q];
            const numerator = fq * (this.k1 + 1);
            const denominator =
                fq +
                this.k1 * (1 - this.b + (this.b * docTokens.length) / this.avgDocLen);

            score += idf * (numerator / denominator);
        }

        return score;
    }

    // --------------------------------------------------------
    // Public: Search
    // --------------------------------------------------------
    public search(query: string, topK = 10): { id: string; score: number }[] {
        const queryTokens = this.tokenizer(query);

        const scores = this.docs.map((doc, idx) => ({
            id: doc.id,
            score: this.scoreOneDoc(queryTokens, this.corpusTokens[idx]),
        }));

        return scores
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }
}
