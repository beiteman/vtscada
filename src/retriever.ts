import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import { PorterStemmer, stopwords } from 'natural';

export type FuncDocument = {
    comments: string[],
    snippets: string[]
}

export type FuncDocumentMap = {
    [key: string]: FuncDocument;
}

export class FuncDocumentRetriever {
    private session: ort.InferenceSession | null = null;
    private vocabulary: { [term: string]: number } | null = null;
    private keys: string[] = []; // function name
    private vocabSize: number = 0;
    private docs: FuncDocumentMap = {};

    // use creator instead
    private constructor() {
    }

    public static async create(
        onnxModelFile: string,
        modelInfoFile: string,
        docFile: string
    ): Promise<FuncDocumentRetriever> {
        const instance = new FuncDocumentRetriever();

        console.log(`[INFO] Loading model info from: ${modelInfoFile}`);
        const docs: FuncDocumentMap = JSON.parse(fs.readFileSync(docFile, 'utf8'));
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

    private analyzeQuery(query: string, useStemming: boolean = true): string[] {
        let qClean = query.replace(/[^\w\s]/g, " ").toLowerCase();
        let tokens = qClean.split(/\s+/).filter(t => t.length > 0 && !stopwords.includes(t));

        if (useStemming) {
            tokens = tokens.map(t => PorterStemmer.stem(t));
        }
        return tokens;
    }

    private vectorizeQuery(queryTokens: string[]): ort.Tensor {
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

    public async retrieve(query: string, topK: number = 5): Promise<FuncDocument[]> {
        if (this.session == null) throw new Error('this.session is null');

        // Preprocess
        const tokens = this.analyzeQuery(query);
        const queryVector = this.vectorizeQuery(tokens);

        // ONNX Inference
        const feeds = { 'query_vector': queryVector };
        const results = await this.session.run(feeds);

        // Process and Rank Results
        const scores = results['similarity_scores'].data as Float32Array;
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