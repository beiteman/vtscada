import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import { FuncDocument, FuncDocumentMap } from './docs';
import { STOPWORDS, porterStem } from './naturalV2';

export class TFIDFRetriever {
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
    ): Promise<TFIDFRetriever> {
        const instance = new TFIDFRetriever();
        const docs: FuncDocumentMap = JSON.parse(fs.readFileSync(docFile, 'utf8'));
        const modelInfo = JSON.parse(fs.readFileSync(modelInfoFile, 'utf8'));
        instance.vocabulary = modelInfo.vocabulary;
        instance.keys = modelInfo.keys;
        instance.docs = docs;
        instance.vocabSize = Object.keys(modelInfo.vocabulary).length;
        instance.session = await ort.InferenceSession.create(onnxModelFile);
        return instance;
    }

    private analyzeQuery(query: string, useStemming: boolean = true): string[] {
        let qClean = query.replace(/[^\w\s]/g, " ").toLowerCase();
        // let tokens = qClean.split(/\s+/).filter(t => t.length > 0 && !stopwords.includes(t));
        let tokens = qClean.split(/\s+/).filter(t => t.length > 0 && !STOPWORDS.has(t));

        if (useStemming) {
            // tokens = tokens.map(t => PorterStemmer.stem(t));
            tokens = tokens.map(t => porterStem(t));
        }
        console.log(query, tokens);
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
        console.log(queryVector);
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

        theResults.forEach(r => console.log(r.key, r.score));

        return theResults.map(r => this.docs[r.key]);
    }

}