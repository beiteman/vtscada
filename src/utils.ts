export function getLeadingWhitespace(str: string): string {
    const match = str.match(/^\s*/);
    return match ? match[0] : '';
}

// export function countNewlines(str: string): number {
//     const matches = str.match(/\n/g);
//     return matches ? matches.length : 0;
// }

// export function addLeadingWhitespaces(str: string, leadingWhitespace: string): string {
//     return str.replace(/(\r?\n)/g, `$1${leadingWhitespace}`);
// }

// // Replaces any sequence of one or more whitespace 
// // characters (spaces, tabs, newlines) with a single space
// export function sanitizeText(text: string): string {
//     return text.replace(/\s+/g, ' ').trim().substring(0, 200);
// }

// /**
//  * Splits a string into an array of lines, ensuring no word is cut in the middle.
//  * Each line aims to be near the target maximum length (n), but may be slightly
//  * shorter or longer to accommodate full words.
//  *
//  * @param text The input string to wrap.
//  * @param n The target maximum line length.
//  * @returns An array of strings, where each string is a wrapped line.
//  */
// export function wordWrap(text: string, n: number): string[] {

//     const lines: string[] = [];
//     let workingText = text.trim();

//     while (workingText.length > 0) {
//         if (workingText.length <= n) {
//             lines.push(workingText);
//             break;
//         }

//         // find the last space within limit
//         let breakIndex = workingText.lastIndexOf(' ', n);

//         if (breakIndex === -1) {
//             // no space found: break at first space after n
//             const nextSpace = workingText.indexOf(' ', n);
//             if (nextSpace === -1) {
//                 lines.push(workingText);
//                 break;
//             }
//             lines.push(workingText.substring(0, nextSpace));
//             workingText = workingText.substring(nextSpace + 1);
//         } else {
//             lines.push(workingText.substring(0, breakIndex));
//             workingText = workingText.substring(breakIndex + 1);
//         }

//         workingText = workingText.trimStart();
//     }

//     return lines;
// }
