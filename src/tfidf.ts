import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import { FuncDocument, FuncDocumentMap } from './docs';

import {stemmer} from 'stemmer'

export function porterStem(word: string): string {
    return stemmer(word);
}

// https://github.com/stdlib-js/datasets-stopwords-en/blob/main/data/words.json
export const STOPWORDS: Set<string> = new Set([
    "a", "about", "above", "across", "actually", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "behind", "being", "best", "better", "between", "both", "but", "by", "c", "came", "can", "certain", "certainly", "clearly", "come", "consider", "considering", "could", "d", "did", "different", "do", "does", "doing", "done", "down", "downwards", "during", "e", "each", "eg", "eight", "either", "enough", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "fact", "facts", "far", "few", "first", "five", "for", "four", "from", "further", "g", "get", "gets", "given", "gives", "go", "going", "got", "h", "had", "has", "have", "having", "he", "her", "here", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "into", "is", "it", "its", "itself", "j", "just", "k", "keep", "keeps", "knew", "know", "known", "knows", "l", "last", "later", "least", "less", "let", "like", "likely", "m", "many", "may", "me", "might", "more", "most", "mostly", "much", "must", "my", "myself", "n", "necessary", "need", "needs", "never", "new", "next", "nine", "no", "nobody", "non", "not", "nothing", "now", "nowhere", "o", "of", "off", "often", "old", "on", "once", "one", "only", "or", "other", "others", "our", "out", "over", "p", "per", "perhaps", "please", "possible", "put", "q", "quite", "r", "rather", "really", "right", "s", "said", "same", "saw", "say", "says", "second", "see", "seem", "seemed", "seems", "seven", "several", "shall", "she", "should", "since", "six", "so", "some", "somebody", "someone", "something", "somewhere", "still", "such", "sure", "t", "take", "taken", "ten", "than", "that", "the", "their", "them", "then", "there", "therefore", "therein", "thereupon", "these", "they", "think", "third", "this", "those", "though", "three", "through", "thus", "to", "together", "too", "took", "toward", "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very", "w", "want", "wanted", "wants", "was", "way", "we", "well", "went", "were", "what", "when", "where", "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within", "without", "would", "x", "y", "yet", "you", "your", "yours", "z"
]);

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