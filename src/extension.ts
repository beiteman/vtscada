import * as vscode from 'vscode';
import * as path from 'path';
import { Range, Position } from 'vscode';
import { FuncDocument } from './docs';
import { TFIDFRetriever } from './tfidf';
import { getLeadingWhitespace } from './utils';
import { Lang, LanguageIdentifier } from './languages';
import { BM25Retriever } from './bm25';

const TOP_K: number = 5;
const DEFAULT_LANG: Lang = 'en';

async function inlineCompletionItems(
    retriever: TFIDFRetriever | BM25Retriever,
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
    documents.forEach(item => {

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
                insertText = line;
            } else {
                // next lines
                insertText = `${insertText}\n${leadingWhitespace}${line}`;
            }
        });

        snippets.forEach((line, _) => {
            insertText = `${insertText}\n${leadingWhitespace}${line}`;
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
    const retrieversTFIDF: Map<Lang, TFIDFRetriever> = new Map();
    const retrieversBM25: Map<Lang, BM25Retriever> = new Map();

    // english (TFIDF)
    const enModelInfoPath = path.join(context.extensionPath, "resources", `info.en.json`);
    const enModelPath = path.join(context.extensionPath, "resources", `model.en.onnx`);
    const enDocsPath = path.join(context.extensionPath, "resources", `docs.en.json`);
    TFIDFRetriever.create(enModelPath, enModelInfoPath, enDocsPath).then(retriever => {
        retrieversTFIDF.set('en', retriever);
        vscode.window.showInformationMessage(`loaded "en"`);
        console.log(`loaded "en"`);
    }).catch(error => {
        vscode.window.showErrorMessage(`Error loading model (en): "${error}"`);
        console.error(error);
        return Promise.reject(error);
    });

    // chinese (traditional)
    const twDocsPath = path.join(context.extensionPath, "resources", `docs.zh-tw.json`);
    BM25Retriever.create(twDocsPath, true).then(retriever => {
        retrieversBM25.set('zh-tw', retriever);
        vscode.window.showInformationMessage(`loaded "zh-tw"`);
        console.log(`loaded "zh-tw"`);
    }).catch(error => {
        vscode.window.showErrorMessage(`Error loading model (zh-tw): "${error}"`);
        console.error(error);
        return Promise.reject(error);
    });

    // chinese (simplified)
    const cnDocsPath = path.join(context.extensionPath, "resources", `docs.zh-cn.json`);
    BM25Retriever.create(cnDocsPath, true).then(retriever => {
        retrieversBM25.set('zh-cn', retriever);
        vscode.window.showInformationMessage(`loaded "zh-cn"`);
        console.log(`loaded "zh-cn"`);
    }).catch(error => {
        vscode.window.showErrorMessage(`Error loading model (zh-cn): "${error}"`);
        console.error(error);
        return Promise.reject(error);
    });

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
                    // check the query language
                    const lang = langIdentifier.identify(query, DEFAULT_LANG);
                    console.log("lang", lang);
                    console.log("query", query);
                    if (lang == 'en') {
                        const retriever = retrieversTFIDF.get('en');
                        if (retriever == undefined) {
                            await vscode.window.showErrorMessage(`still loading "${lang}" ...`);
                        } else {
                            let items = await inlineCompletionItems(retriever, document, position, query);
                            return { items: items };
                        }
                    } else {
                        const retriever = retrieversBM25.get(lang);
                        if (retriever == undefined) {
                            await vscode.window.showErrorMessage(`still loading "${lang}" ...`);
                        } else {
                            let items = await inlineCompletionItems(retriever, document, position, query);
                            return { items: items };
                        }
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

