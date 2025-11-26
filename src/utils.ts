export function getLeadingWhitespace(str: string): string {
    const match = str.match(/^\s*/);
    return match ? match[0] : '';
}
