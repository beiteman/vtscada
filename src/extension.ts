import * as vscode from 'vscode';
import * as net from 'net';
import { Range, Position } from 'vscode';


const SERVER_HOST = "127.0.0.1";
const SERVER_PORT = 3000;

interface ServerResponse {
    items: {
        insertText: string;
        insertTextFormat: null | number;
        range: {
            start: { line: number; character: number };
            end: { line: number; character: number };
        };
        command?: {
            command: string;
            title: string;
            arguments: any[];
        };
    }[];
}

export function activate(context: vscode.ExtensionContext) {
    console.log('Inline completions TCP client activated');

    const client = new net.Socket();
    let isConnected = false;
    let responseBuffer = '';

    // Connect to Python server
    client.connect(SERVER_PORT, SERVER_HOST, () => {
        isConnected = true;
        console.log('Connected to Python server');
    });

    client.on('data', (data) => {
        responseBuffer += data.toString();
        // Check if we've received a complete JSON message (ends with newline)
        if (responseBuffer.includes('\n')) {
            const messages = responseBuffer.split('\n');
            responseBuffer = messages.pop() || ''; // Save incomplete part
        }
    });

    client.on('error', (err) => {
        console.error('Connection error:', err);
        isConnected = false;
    });

    client.on('close', () => {
        isConnected = false;
        console.log('Connection closed');
    });

    // Register command
    vscode.commands.registerCommand('vtscada.query', async (...args) => {
        vscode.window.showInformationMessage('Completion accepted: ' + JSON.stringify(args));
    });

    const provider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position) {
            if (!isConnected) return { items: [] };

            try {
                // Prepare request with full document and position
                const request = {
                    document: {
                        text: document.getText(),
                        uri: document.uri.toString(),
                        languageId: document.languageId,
                        lineCount: document.lineCount
                    },
                    position: {
                        line: position.line,
                        character: position.character
                    },
                    context: {
                        triggerKind: 0 // Invoked automatically
                    }
                };

                // Send request and wait for response
                const response = await new Promise<ServerResponse>((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        reject(new Error('Server timeout after 2 seconds'));
                    }, 2000);

                    client.write(JSON.stringify(request) + '\n', (err) => {
                        if (err) reject(err);
                    });

                    const listener = (data: Buffer) => {
                        try {
                            const message = data.toString();
                            if (message.trim()) {
                                clearTimeout(timeout);
                                client.off('data', listener);
                                resolve(JSON.parse(message));
                            }
                        } catch (e) {
                            reject(e);
                        }
                    };

                    client.on('data', listener);
                });

                // Convert server response to VS Code completion items
                const completionItems = response.items.map(item => {
                    const range = new Range(
                        new Position(item.range.start.line, item.range.start.character),
                        new Position(item.range.end.line, item.range.end.character)
                    );

                    return {
						label: "label",
						detail: "detail",
						documentation: "documentation",
						kind: vscode.CompletionItemKind.Function,
                        insertText: new vscode.SnippetString(item.insertText),
                        insertTextFormat: item.insertTextFormat,
                        range: range,
                        command: item.command ? {
                            command: item.command.command,
                            title: item.command.title,
                            arguments: item.command.arguments
                        } : undefined
                    };
                });

                return {
                    items: completionItems
                };

            } catch (error) {
                console.error('Error getting completions:', error);
                return { items: [] };
            }
        }
    };

    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);

    // Clean up on deactivation
    context.subscriptions.push({
        dispose: () => {
            if (isConnected) {
                client.end();
            }
        }
    });
}

export function deactivate() {
    // Cleanup handled by subscription
}