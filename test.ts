import * as path from 'path';
import { STRetriever } from './src/transformer';
import { readFileSync, writeFileSync } from 'fs';

const RES_ROOT = path.join(".", "resource");
const INDEX_EN = path.join(RES_ROOT, "index.en.json");
const INDEX_CN = path.join(RES_ROOT, "index.zh-cn.json");
const INDEX_TW = path.join(RES_ROOT, "index.zh-tw.json");
const MODEL_ROOT_EN = path.join(RES_ROOT, "all-MiniLM-L6-v2");
const MODEL_ROOT_CN = path.join(RES_ROOT, "text2vec-base-chinese");
const MODEL_ROOT_TW = path.join(RES_ROOT, "text2vec-base-chinese");
const TESTCASE_FILEPATH = path.join("./tests", "testcases.json");
const RESULT_FILEPATH = path.join("./tests", "test-results.txt");

interface TestCase {
    key: string;
    query: string;
    lang: string;
}

function loadTestCases(filePath: string = './testcases.json'): TestCase[] {
    try {
        const data: TestCase[] = JSON.parse(readFileSync(filePath, 'utf-8'));
        return data;
    } catch (error) {
        console.error("Error loading test cases:", error);
        return [];
    }
}

async function runTests() {
    try {
        const testCases = loadTestCases(TESTCASE_FILEPATH);
        const retrievers: Record<string, any> = {
            'en': await STRetriever.createAndLoad(
                path.join(MODEL_ROOT_EN, "model.onnx"),
                path.join(MODEL_ROOT_EN, "tokenizer.json"),
                path.join(MODEL_ROOT_EN, "tokenizer_config.json"),
                INDEX_EN),
            'zh-cn': await STRetriever.createAndLoad(
                path.join(MODEL_ROOT_CN, "model.onnx"),
                path.join(MODEL_ROOT_CN, "tokenizer.json"),
                path.join(MODEL_ROOT_CN, "tokenizer_config.json"),
                INDEX_CN),
            'zh-tw': await STRetriever.createAndLoad(
                path.join(MODEL_ROOT_TW, "model.onnx"),
                path.join(MODEL_ROOT_TW, "tokenizer.json"),
                path.join(MODEL_ROOT_TW, "tokenizer_config.json"),
                INDEX_TW)
        };

        // Record<Lang, Record<Key, Score[]>
        const groupedData: Record<string, Record<string, { query: string, score: number }[]>> = {};
        await Promise.all(testCases.map(async (test) => {
            const retriever = retrievers[test.lang];
            if (!retriever) return;
            const score = await retriever.retrieveForTest(test.query, 5, test.key);
            console.log(`Testing lang=${test.lang} query=${test.query} score=${score}`);
            if (!groupedData[test.lang]) groupedData[test.lang] = {};
            if (!groupedData[test.lang][test.key]) groupedData[test.lang][test.key] = [];
            groupedData[test.lang][test.key].push({ query: test.query, score });
        }));

        // 2. Build the text report string
        let report = `TEST RESULTS - ${new Date().toLocaleString()}\n`;
        report += `==========================================\n\n`;
        for (const [lang, keys] of Object.entries(groupedData)) {
            report += `LANGUAGE: ${lang.toUpperCase()}\n`;
            report += `------------------------------------------\n`;
            const keyAverages: number[] = [];
            for (const [key, results] of Object.entries(keys)) {
                const avgKeyScore = results.reduce((a, b) => a + b.score, 0) / results.length;
                keyAverages.push(avgKeyScore);
                report += `Key: ${key} (Avg: ${avgKeyScore.toFixed(4)})\n`;
                results.forEach(res => {
                    report += `  - [Score: ${res.score.toString().padEnd(6)}] Query: ${res.query}\n`;
                });
                report += `\n`;
            }

            const langTotalAvg = keyAverages.reduce((a, b) => a + b, 0) / keyAverages.length;
            report += `>> TOTAL AVERAGE FOR ${lang.toUpperCase()}: ${langTotalAvg.toFixed(4)}\n`;
            report += `==========================================\n\n`;
        }
        writeFileSync(RESULT_FILEPATH, report, 'utf-8');
        console.log(`Detailed report saved to ${RESULT_FILEPATH}`);
    } catch (error) {
        console.error("Test execution failed:", error);
    }
}

runTests().catch(console.error);
