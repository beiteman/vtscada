import * as vscode from 'vscode';
import * as path from 'path';
import { Range, Position } from 'vscode';
import { FuncDocument } from './docs';
import { TFIDFRetriever } from './tfidf';
import { STRetriever } from './transformerV2';
import { getLeadingWhitespace } from './utils';
import { Lang, LanguageIdentifier } from './languagesV2';
import { BM25Retriever } from './bm25';

const TOP_K: number = 5;

function sanitize(text: string): string {
    const clean = text.normalize("NFC");
    return clean;
}

async function inlineCompletionItems(
    retriever: TFIDFRetriever | BM25Retriever | STRetriever,
    document: vscode.TextDocument,
    position: vscode.Position,
    query: string) {

    const character = position.character > 0 ? position.character : 1;
    const range = new Range(
        new Position(position.line, character),
        new Position(position.line, character)
    );

    let leadingWhitespace = getLeadingWhitespace(document.lineAt(position.line - 1).text);
    const result: vscode.InlineCompletionItem[] = [];
    const documents: FuncDocument[] = await retriever.retrieve(query, TOP_K);
    documents.forEach((item, index) => {

        const comments = item.comments;
        const snippets = item.snippets;

        const commentsRange = new vscode.Range(
            new vscode.Position(position.line, 0),
            new vscode.Position(position.line + comments.length, 0)
        );

        let insertText: string = "";
        comments.forEach((line, index) => {
            if (index == 0) {
                // first line
                insertText = sanitize(line);
            } else {
                // next lines
                insertText = `${insertText}\n${leadingWhitespace}${sanitize(line)}`;
            }
        });

        snippets.forEach((line, _) => {
            insertText = `${insertText}\n${leadingWhitespace}${sanitize(line)}`;
        });

        result.push({
            insertText: insertText,
            range: range,
            command: {
                command: 'extension.removeLines',
                title: 'Cleanup docs after inline completion',
                arguments: [commentsRange]
            },
        })
    });
    return result;

}

export function activate(context: vscode.ExtensionContext) {

    // model init --------------------------
    const retrieversST: Map<Lang, STRetriever> = new Map();

    const resPath = path.join(context.extensionPath, "resources", "data");

    // chinese
    {
        const lang = 'zh-tw';
        const basePath = path.join(resPath, "text2vec-base-chinese");
        const onnxPath = path.join(basePath, "model.onnx");
        const tokenizerPath = path.join(basePath, "tokenizer.json");
        const tokenizerConfigPath = path.join(basePath, "tokenizer_config.json");
        const indexPath = path.join(resPath, `index.${lang}.json`);
        STRetriever.createAndLoad(onnxPath, tokenizerPath, tokenizerConfigPath, indexPath).then(retriever => {
            retrieversST.set(lang, retriever);
            console.log("Loaded!");
        }).catch(error => {
            vscode.window.showErrorMessage(`Error ${basePath}: "${error}"`);
            console.error(error);
            return Promise.reject(error);
        });
    }

    {
        const lang = 'zh-cn';
        const basePath = path.join(resPath, "text2vec-base-chinese");
        const onnxPath = path.join(basePath, "model.onnx");
        const tokenizerPath = path.join(basePath, "tokenizer.json");
        const tokenizerConfigPath = path.join(basePath, "tokenizer_config.json");
        const indexPath = path.join(resPath, `index.${lang}.json`);
        STRetriever.createAndLoad(onnxPath, tokenizerPath, tokenizerConfigPath, indexPath).then(retriever => {
            retrieversST.set(lang, retriever);
            console.log("Loaded!");
        }).catch(error => {
            vscode.window.showErrorMessage(`Error ${basePath}: "${error}"`);
            console.error(error);
            return Promise.reject(error);
        });
    }

    // english
    {
        const lang = 'en';
        const basePath = path.join(resPath, "all-MiniLM-L6-v2");
        const onnxPath = path.join(basePath, "model.onnx");
        const tokenizerPath = path.join(basePath, "tokenizer.json");
        const tokenizerConfigPath = path.join(basePath, "tokenizer_config.json");
        const indexPath = path.join(resPath, `index.${lang}.json`);
        STRetriever.createAndLoad(onnxPath, tokenizerPath, tokenizerConfigPath, indexPath).then(retriever => {
            retrieversST.set(lang, retriever);
            console.log("Loaded!");
        }).catch(error => {
            vscode.window.showErrorMessage(`Error ${basePath}: "${error}"`);
            console.error(error);
            return Promise.reject(error);
        });
    }

    // language identifier --------------------------
    const langIdentifier = new LanguageIdentifier();
    // ----------------------------------------------

    const provider: vscode.InlineCompletionItemProvider = {

        async provideInlineCompletionItems(document, position, context, _token) {
            const regexp = /^(\s*)\{\s*(.*?)\s*\}\s*$/;
            if (position.line <= 0) return { items: [] };
            const lineBefore = document.lineAt(position.line - 1).text;
            const match = lineBefore.match(regexp);
            if (match) {
                const query = match[2].trim();
                if (!query) return { items: [] };
                try {
                    let lang = langIdentifier.identify(query);
                    if (lang == 'Undetermined') {
                        console.log("Undetermined language for query", query);
                        lang = 'en'; // default
                    }
                    const retriever = retrieversST.get(lang);
                    if (retriever == undefined) {
                        await vscode.window.showErrorMessage(`still loading "${lang}" ...`);
                    } else {
                        console.log(`using model language: ${lang}`);
                        let items = await inlineCompletionItems(retriever, document, position, query);
                        return { items: items };
                    }
                } catch (error) {
                    console.error(error);
                    await vscode.window.showErrorMessage(`Error: "${error}"`);
                }
            }
            return { items: [] };
        }
    };
    // context.subscriptions.push(
    //     vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider)
    // );
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ language: 'vtscadascript' }, provider)
    );

    // --- Clean up comment after accepting ---
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'extension.removeLines',
            async (range: vscode.Range) => {
                const editor = vscode.window.activeTextEditor;
                if (!editor) return;

                await editor.edit(editBuilder => {
                    editBuilder.delete(range);
                });
            })
    );

}

