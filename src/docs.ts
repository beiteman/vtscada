export type FuncDocument = {
    comments: string[],
    snippets: string[],
    key: string | null | undefined,
}

export type FuncDocumentMap = {
    [key: string]: FuncDocument;
}