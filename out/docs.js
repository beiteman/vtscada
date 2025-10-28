"use strict";
// docs.append({
//             "name": name,
//             "description": {
//                 "en": description_map.get(name, {"en": ""}).get("en"),
//                 "zh-cn": description_map.get(name, {"zh-cn": ""}).get("zh-cn"),
//                 "zh-tw": description_map.get(name, {"zh-tw": ""}).get("zh-tw")
//             },
//             "comments": {
//                 "en": comments_map.get(name, {"en": ""}).get("en"),
//                 "zh-cn": comments_map.get(name, {"zh-cn": ""}).get("zh-cn"),
//                 "zh-tw": comments_map.get(name, {"zh-tw": ""}).get("zh-tw")
//             },
//             "snippets": snippets_map.get(name, {"function": "", "params": []}),
//             "usage": usage_map.get(name, {"steady_state": False, "script": False})
//         })
Object.defineProperty(exports, "__esModule", { value: true });
// import { WORD_WRAP_LENGTH } from "./configs";
// import { sanitizeText, wordWrap } from "./utils";
// export type Doc = {
//     name?: string;
//     url?: string;
//     description?: string;
//     returns?: string;
//     usage?: string;
//     function_groups?: string;
//     related_to?: string;
//     format?: string;
//     parameters: { name: string, description: string }[];
//     comments?: string;
//     example?: string;
// }
// export type FunctionUsage = "script" | "steady state" | "both" | "unknown"
// const script: FunctionUsage = "script";
// const steadyState: FunctionUsage = "steady state";
// const both: FunctionUsage = "both";
// const unknown: FunctionUsage = "unknown";
// export function isSteadyStateFunction(usage?: string): boolean {
//     const _usage = getFunctionUsage(usage);
//     return (_usage === both || _usage == steadyState);
// }
// export function getFunctionUsage(usage?: string): FunctionUsage {
//     if (usage == undefined) {
//         return unknown;
//     }
//     const _usage = usage.toLowerCase();
//     if (_usage.includes(script) && _usage.includes("steady")) {
//         return both;
//     } else if (_usage.includes(script)) {
//         return script;
//     } else if (_usage.includes("steady")) {
//         return steadyState;
//     } else {
//         return unknown;
//     }
// }
// // parse docs into "comments" and "snippets" (multiple lines)
// export function parseDoc(doc: Doc): [string[], string[]] {
//     // extract comments
//     // --- first priority: description
//     let comments: string[] = [];
//     if (doc.description != undefined) {
//         if (doc.description.trim() !== "") {
//             const text = `{ DESCRIPTION: ${sanitizeText(doc.description)} }`;
//             wordWrap(text, WORD_WRAP_LENGTH).forEach((str, idx) => {
//                 if (idx == 0) comments.push(str)
//                 else comments.push(`\t${str}`)
//             });
//         }
//     }
//     // if (doc.name != undefined) {
//     //     if (doc.name.trim() !== "") {
//     //         comments.push(`{ NAME: ${doc.name.trim()} }`);
//     //     }
//     // }
//     // if (doc.format != undefined) {
//     //     if (doc.format.trim() !== "") {
//     //         comments.push(`{ FORMAT: ${doc.format.trim()} }`);
//     //     }
//     // }
//     // // --- second priority: comment
//     // if (doc.comments !== undefined) {
//     //     if (doc.comments.trim() !== "") {
//     //         const text = `{ ${sanitizeText(doc.comments)} }`;
//     //         comments.push(...wordWrap(text, WORD_WRAP_LENGTH));
//     //     }
//     // }
//     // extract code
//     const snippets: string[] = [];
//     if (doc.name != undefined) {
//         if (doc.name.trim() !== "") {
//             if (doc.format != undefined) {
//                 const name = `${doc.name.trim()}(`;
//                 const index = doc.format.indexOf(name);
//                 const header = `${doc.format.substring(0, index)}${name}`;
//                 if (doc.parameters.length > 0) {
//                     snippets.push(header.trim());
//                     const paramCount = doc.parameters.length;
//                     doc.parameters.forEach((param, index) => {
//                         const paramName = param.name.trim();
//                         const paramDesc = param.description.trim();
//                         if (index == paramCount - 1) { // last param
//                             const text = `${paramName} { ${sanitizeText(paramDesc)} }`;
//                             wordWrap(text, WORD_WRAP_LENGTH).forEach(str => {
//                                 snippets.push(`\t${str}`)
//                             });
//                             snippets.push(");")
//                         } else {
//                             const text = `${paramName}, { ${sanitizeText(paramDesc)} }`;
//                             wordWrap(text, WORD_WRAP_LENGTH).forEach(str => {
//                                 snippets.push(`\t${str}`)
//                             });
//                         }
//                     });
//                 } else {
//                     snippets.push(`${header.trim()}();`);
//                 }
//             }
//         }
//     }
//     return [comments, snippets]
// }
