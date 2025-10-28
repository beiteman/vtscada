"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const vscode_1 = require("vscode");
const retriever_1 = require("./retriever");
const utils_1 = require("./utils");
const languages_1 = require("./languages");
const defaultLang = 'en';
const recognizedLangs = ['en', 'zh-tw', 'zh-cn'];
const topK = 5;
async function getInlineCompletionItems(retriever, document, position, query) {
    const character = position.character > 0 ? position.character : 1;
    const range = new vscode_1.Range(new vscode_1.Position(position.line, character), new vscode_1.Position(position.line, character));
    let leadingWhitespace = (0, utils_1.getLeadingWhitespace)(document.lineAt(position.line - 1).text);
    const result = [];
    const documents = await retriever.retrieve(query, topK);
    documents.forEach(item => {
        const comments = item.comments;
        const snippets = item.snippets;
        const commentsRange = new vscode.Range(new vscode.Position(position.line, 0), new vscode.Position(position.line + comments.length, 0));
        let insertText = "";
        comments.forEach((line, index) => {
            if (index == 0) {
                // first line
                insertText = line;
            }
            else {
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
        });
    });
    return result;
}
function activate(context) {
    // model init --------------------------
    const retrieverByLang = new Map();
    const languages = ["en", "zh-tw", "zh-cn"];
    languages.forEach(lang => {
        const modelInfoPath = path.join(context.extensionPath, "resources", `info.${lang}.json`);
        const modelPath = path.join(context.extensionPath, "resources", `model.${lang}.onnx`);
        const docsPath = path.join(context.extensionPath, "resources", `docs.${lang}.json`);
        retriever_1.FuncDocumentRetriever.create(modelPath, modelInfoPath, docsPath)
            .then(retriever => retrieverByLang.set(lang, retriever))
            .catch(error => {
            console.error(`FuncDocumentRetriever(${lang})`, error);
            return Promise.reject(error);
        });
    });
    // language identifier --------------------------
    const langIdentifier = new languages_1.LanguageIdentifier();
    // ----------------------------------------------
    const provider = {
        async provideInlineCompletionItems(document, position, context, _token) {
            const regexp = /^(\s*)\{\s*(.*?)\s*\}\s*$/;
            if (position.line <= 0)
                return { items: [] };
            const lineBefore = document.lineAt(position.line - 1).text;
            const match = lineBefore.match(regexp);
            if (match) {
                const query = match[2].trim();
                if (!query)
                    return { items: [] };
                try {
                    // check the query language
                    const lang = langIdentifier.identify(query, defaultLang, recognizedLangs);
                    const retriever = retrieverByLang.get(lang);
                    if (retriever !== undefined) {
                        let items = await getInlineCompletionItems(retriever, document, position, query);
                        return { items: items };
                    }
                    else {
                        console.warn(`Model not found "${lang}"`);
                    }
                }
                catch (error) {
                    console.log(error);
                }
            }
            return { items: [] };
        }
    };
    context.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider));
    // --- Clean up comment after accepting ---
    context.subscriptions.push(vscode.commands.registerCommand('extension.removeLines', async (range) => {
        const editor = vscode.window.activeTextEditor;
        if (!editor)
            return;
        await editor.edit(editBuilder => {
            editBuilder.delete(range);
        });
    }));
}
