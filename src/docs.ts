export type FuncDocument = {
    comments: string[],
    snippets: string[]
}

export type FuncDocumentMap = {
    [key: string]: FuncDocument;
}