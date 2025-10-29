import * as vscode from 'vscode';
import * as path from 'path';
import { Range, Position } from 'vscode';
import { FuncDocument, FuncDocumentRetriever } from './retriever';
import { getLeadingWhitespace } from './utils';
import { Lang, LanguageIdentifier } from './languages';

// const languages: Lang[] = ["en", "zh-tw", "zh-cn"];
const languages: Lang[] = ["en"];
const defaultLang: Lang = 'en';
const topK: number = 5;

async function getInlineCompletionItems(
    retriever: FuncDocumentRetriever,
    document: vscode.TextDocument,
    position: vscode.Position,
    query: string): Promise<vscode.InlineCompletionItem[]> {

    const character = position.character > 0 ? position.character : 1;
    const range = new Range(
        new Position(position.line, character),
        new Position(position.line, character)
    );

    let leadingWhitespace = getLeadingWhitespace(document.lineAt(position.line - 1).text);
    const result: vscode.InlineCompletionItem[] = [];
    const documents: FuncDocument[] = await retriever.retrieve(query, topK);
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
    const retrieverByLang: Map<Lang, FuncDocumentRetriever> = new Map();
    const loadedRetrievers: Map<Lang, boolean> = new Map();
    languages.forEach(lang => {
        loadedRetrievers.set(lang, false);
    });
    languages.forEach(lang => {
        const modelInfoPath = path.join(context.extensionPath, "resources", `info.${lang}.json`);
        const modelPath = path.join(context.extensionPath, "resources", `model.${lang}.onnx`);
        const docsPath = path.join(context.extensionPath, "resources", `docs.${lang}.json`);
        FuncDocumentRetriever.create(modelPath, modelInfoPath, docsPath)
            .then(retriever => {
                retrieverByLang.set(lang, retriever);
                loadedRetrievers.set(lang, true);
                vscode.window.showInformationMessage(`Model is loaded "${lang}"`);
            })
            .catch(error => {
                vscode.window.showErrorMessage(`Error loading model (${lang}): "${error}"`);
                return Promise.reject(error);
            });
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
                    const lang = langIdentifier.identify(
                        query, defaultLang, languages);
                    const retriever = retrieverByLang.get(lang);
                    const isLoaded = loadedRetrievers.get(lang);
                    if (retriever !== undefined) {
                        let items = await getInlineCompletionItems(
                            retriever, document, position, query);
                        return { items: items };
                    } else if (isLoaded) {
                        await vscode.window.showErrorMessage(`Model not found "${lang}"`);
                    } else {
                        await vscode.window.showInformationMessage("Still loading ...");
                    }
                } catch (error) {
                    await vscode.window.showErrorMessage(`Error: "${error}"`);
                }
            }
            return { items: [] };
        }
    };
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider)
    );

    // --- Clean up comment after accepting ---
    context.subscriptions.push(
        vscode.commands.registerCommand('extension.removeLines', async (range: vscode.Range) => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            await editor.edit(editBuilder => {
                editBuilder.delete(range);
            });
        })
    );

}

