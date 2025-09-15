import * as net from "net";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  StreamInfo,
  ExecuteCommandRequest
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext) {
  // ---- Connect to remote TCP server ----
  const serverHost = "0.tcp.jp.ngrok.io";
  const serverPort = 12477;

  // SERVER_HOST = "0.tcp.jp.ngrok.io"
  // SERVER_PORT = 12477

  const serverOptions = () => {
    const socket = net.connect(serverPort, serverHost);
    const result: StreamInfo = {
      writer: socket,
      reader: socket
    };
    return Promise.resolve(result);
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file" }, { scheme: "untitled" }],
    synchronize: {}
  };

  client = new LanguageClient(
    "pygen-lsp-client",
    "PyGen LSP Client",
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(client);
  client.start();

  // ---- Command for prompt → server → insert code ----
  const disposable = vscode.commands.registerCommand("pygen.generateCode", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage("Open a file and place the cursor where you want code inserted.");
      return;
    }

    const prompt = await vscode.window.showInputBox({
      prompt: "Describe the code to generate",
      placeHolder: "e.g., create an HTTP handler that prints 'hello'"
    });
    if (!prompt) return;

    const pos = editor.selection.active;
    const uri = editor.document.uri.toString();

    try {
      const result = await client!.sendRequest(ExecuteCommandRequest.type, {
        command: "generateCode",
        arguments: [uri, { line: pos.line, character: pos.character }, prompt]
      });
      if (result) {
        vscode.window.showInformationMessage(String(result));
      }
    } catch (err: any) {
      vscode.window.showErrorMessage(`generateCode failed: ${err?.message ?? String(err)}`);
    }
  });

  context.subscriptions.push(disposable);
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
