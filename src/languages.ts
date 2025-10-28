export type Lang = 'en' | 'zh-cn' | 'zh-tw' | 'Undetermined';
export class LanguageIdentifier {
    // Threshold: If Latin characters make up more than this percentage of the string, 
    // it is classified as English.
    private static readonly LATIN_THRESHOLD = 0.70;

    // Distinguishing Characters: A small, non-exhaustive set of common characters 
    // that have different forms in SC vs TC. Used for heuristic classification.

    // SC Forms (e.g., '体' vs '體', '国' vs '國', '书' vs '書')
    private static readonly SIMPLIFIED_CHARS = new Set("体书风长门东华乐");

    // TC Forms (e.g., '體' vs '体', '國' vs '国', '書' vs '书')
    private static readonly TRADITIONAL_CHARS = new Set("體書風長門東華樂");

    public identify(text: string, defaultLang: Lang, 
        recognizedLangs: string[] = ['en','zh-cn','zh-tw']
    ): Lang {
        const lang = this._identify(text);
        if (recognizedLangs.includes(lang)){
            return lang;
        } else {
            return defaultLang;
        }
    }

    /**
     * Identifies the dominant script used in a string.
     * @param text The input string to analyze.
     * @returns The identified LanguageId.
     */
    private _identify(text: string): Lang {
        if (!text || text.trim().length === 0) {
            return 'Undetermined';
        }

        let totalVisibleChars = 0;
        let latinCount = 0;
        let cjkCount = 0;
        let simplifiedBias = 0;
        let traditionalBias = 0;

        // 1. Initial Pass: Count Latin and CJK characters
        for (const char of text) {
            // Ignore whitespace, newlines, and common punctuation for the total count
            if (/\s|[.,:;!?'"()\[\]{}\-—]/.test(char)) {
                continue;
            }
            totalVisibleChars++;

            // Unicode range for basic Latin letters and numbers (ASCII)
            // \u0000 to \u007F covers most English characters and symbols
            if (char.match(/[\u0000-\u007F]/)) {
                latinCount++;
            }
            // Unicode range for CJK Unified Ideographs (main Chinese characters)
            else if (char.match(/[\u4E00-\u9FFF]/)) {
                cjkCount++;

                // 2. SC/TC Bias Check (only check CJK characters)
                if (LanguageIdentifier.SIMPLIFIED_CHARS.has(char)) {
                    simplifiedBias++;
                } else if (LanguageIdentifier.TRADITIONAL_CHARS.has(char)) {
                    traditionalBias++;
                }
            }
            // Other characters (e.g., Japanese Kana, Korean Hangul, less common CJK extensions) are ignored for classification bias
        }

        // 3. Classification Decision

        // A. English Check: High ratio of Latin characters
        const latinRatio = totalVisibleChars > 0 ? latinCount / totalVisibleChars : 0;
        if (latinRatio >= LanguageIdentifier.LATIN_THRESHOLD) {
            return 'en';
        }

        // B. Chinese Check: High CJK count and low Latin ratio
        if (cjkCount > 0) {
            if (simplifiedBias > traditionalBias * 1.5) {
                // Significantly more simplified characters found
                return 'zh-cn';
            } else if (traditionalBias > simplifiedBias * 1.5) {
                // Significantly more traditional characters found
                return 'zh-tw';
            } else if (simplifiedBias + traditionalBias > 0) {
                // Not enough distinguishing characters to make a clear call, but leaning towards Chinese
                // Fallback to the higher count or default to Simplified if counts are close
                return simplifiedBias >= traditionalBias ? 'zh-cn' : 'zh-tw';
            } else {
                // String is CJK but contains none of the specific distinguishing characters.
                // Default to Simplified Chinese as it's often the modern default, or 'Mixed'.
                // Returning 'Simplified Chinese' for generality here.
                return 'zh-cn';
            }
        }

        // C. Fallback
        return 'Undetermined';
    }
}