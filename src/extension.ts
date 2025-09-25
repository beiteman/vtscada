import * as vscode from 'vscode';
import * as net from 'net';
import { Range, Position } from 'vscode';

// // tcp (ngrok)
// const SERVER_HOST = "0.tcp.jp.ngrok.io";
// const SERVER_PORT = 12090;

// localhost
const SERVER_HOST = "127.0.0.1"
const SERVER_PORT = 2087;

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
        console.log(`Connected to Python server ${SERVER_HOST}:${SERVER_PORT}`);
    });

    client.on('data', (data) => {
        responseBuffer += data.toString();
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
		console.log('Completion accepted: ' + JSON.stringify(args));
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
                    }, 5000);

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
							console.log("error " + e);
                            reject(e);
                        }
                    };

                    client.on('data', listener);
                });

                // Convert server response to VS Code completion items
                const completionItems = [];
				for (const item of response.items) {
					try {
						const range = new Range(
							new Position(item.range.start.line, item.range.start.character),
							new Position(item.range.end.line, item.range.end.character)
						);

						const NEW_LINE_ENCODED = "<NEWLINE>";

						// decode new line
						var insertText = item.insertText.replaceAll(NEW_LINE_ENCODED, '\n');

						// remove non ascii character to prevent json error
						insertText = insertText.replace(/[^\x00-\x7F]/g, "");

						// If all parsing/creation steps succeed, push the item
						completionItems.push({
							label: "label",
							detail: "detail",
							documentation: "documentation",
							kind: vscode.CompletionItemKind.Function,
							insertText: new vscode.SnippetString(insertText),
							insertTextFormat: item.insertTextFormat,
							range: range,
							command: item.command ? {
								command: item.command?.command,
								title: item.command.title,
								arguments: item.command.arguments
							} : undefined
						});

					} catch (error) {
						console.error("Failed to process completion item:", error, item);
					}
				}

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