// remove: import { PorterStemmer, stopwords } from 'natural';

// ------- Lightweight stopwords list (English) -------
export const STOPWORDS: Set<string> = new Set([
  "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
  "be","because","been","before","being","below","between","both","but","by",
  "can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during",
  "each","few","for","from","further",
  "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's",
  "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
  "let's","me","more","most","mustn't","my","myself",
  "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
  "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
  "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too",
  "under","until","up",
  "very",
  "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
  "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
]);

// ------- Compact Porter Stemmer (JS) -------
// This is a compact but functional port of Porter's algorithm.
// Source idea: common small JS implementations of the Porter stemmer.
export function porterStem(word: string): string {
  if (word.length < 3) return word;

  const isConsonant = (w: string, i: number): boolean => {
    const ch = w[i];
    if ('aeiou'.includes(ch)) return false;
    if (ch === 'y') return i === 0 ? true : !isConsonant(w, i - 1);
    return true;
  };

  const measure = (w: string) => {
    let m = 0;
    let i = 0;
    const L = w.length;
    while (i < L) {
      while (i < L && !isConsonant(w, i)) i++;
      if (i >= L) break;
      while (i < L && isConsonant(w, i)) i++;
      m++;
    }
    return m;
  };

  const containsVowel = (w: string) => {
    for (let i = 0; i < w.length; i++) {
      if (!isConsonant(w, i)) return true;
    }
    return false;
  };

  const endsWithDoubleConsonant = (w: string) => {
    const L = w.length;
    if (L < 2) return false;
    return w[L - 1] === w[L - 2] && isConsonant(w, L - 1);
  };

  const cvc = (w: string) => {
    if (w.length < 3) return false;
    const L = w.length;
    return isConsonant(w, L - 1) && !isConsonant(w, L - 2) && isConsonant(w, L - 3) &&
      !'wxy'.includes(w[L - 1]);
  };

  let stem = word;

  // Step 1a
  if (stem.endsWith('sses')) stem = stem.slice(0, -2);
  else if (stem.endsWith('ies')) stem = stem.slice(0, -2);
  else if (stem.endsWith('ss')) {}
  else if (stem.endsWith('s')) stem = stem.slice(0, -1);

  // Step 1b
  let flag = false;
  if (stem.endsWith('eed')) {
    const base = stem.slice(0, -3);
    if (measure(base) > 0) stem = stem.slice(0, -1);
  } else if ((stem.endsWith('ed') && containsVowel(stem.slice(0, -2))) ||
             (stem.endsWith('ing') && containsVowel(stem.slice(0, -3)))) {
    if (stem.endsWith('ed')) stem = stem.slice(0, -2);
    else stem = stem.slice(0, -3);
    flag = true;
    if (stem.endsWith('at') || stem.endsWith('bl') || stem.endsWith('iz')) {
      stem = stem + 'e';
    } else if (endsWithDoubleConsonant(stem) && !['l', 's', 'z'].includes(stem[stem.length - 1])) {
      stem = stem.slice(0, -1);
    } else if (measure(stem) === 1 && cvc(stem)) {
      stem = stem + 'e';
    }
  }

  // Step 1c
  if (!flag && stem.endsWith('y') && containsVowel(stem.slice(0, -1))) {
    stem = stem.slice(0, -1) + 'i';
  }

  // Step 2 - map double suffices to single ones
  const step2list: { [k: string]: string } = {
    'ational': 'ate',
    'tional': 'tion',
    'enci': 'ence',
    'anci': 'ance',
    'izer': 'ize',
    'abli': 'able',
    'alli': 'al',
    'entli': 'ent',
    'eli': 'e',
    'ousli': 'ous',
    'ization': 'ize',
    'ation': 'ate',
    'ator': 'ate',
    'alism': 'al',
    'iveness': 'ive',
    'fulness': 'ful',
    'ousness': 'ous',
    'aliti': 'al',
    'iviti': 'ive',
    'biliti': 'ble'
  };
  for (const [suf, rep] of Object.entries(step2list)) {
    if (stem.endsWith(suf)) {
      const base = stem.slice(0, -suf.length);
      if (measure(base) > 0) {
        stem = base + rep;
      }
      break;
    }
  }

  // Step 3
  const step3list: { [k: string]: string } = {
    'icate': 'ic',
    'ative': '',
    'alize': 'al',
    'iciti': 'ic',
    'ical': 'ic',
    'ful': '',
    'ness': ''
  };
  for (const [suf, rep] of Object.entries(step3list)) {
    if (stem.endsWith(suf)) {
      const base = stem.slice(0, -suf.length);
      if (measure(base) > 0) {
        stem = base + rep;
      }
      break;
    }
  }

  // Step 4
  const step4list = ['al','ance','ence','er','ic','able','ible','ant','ement','ment','ent','ion','ou','ism','ate','iti','ous','ive','ize'];
  for (const suf of step4list) {
    if (stem.endsWith(suf)) {
      const base = stem.slice(0, -suf.length);
      if (measure(base) > 1) {
        if (suf === 'ion') {
          // keep only if preceding char is s or t
          const ch = base[base.length - 1];
          if (ch === 's' || ch === 't') stem = base;
        } else {
          stem = base;
        }
      }
      break;
    }
  }

  // Step 5a
  if (stem.endsWith('e')) {
    const base = stem.slice(0, -1);
    if (measure(base) > 1 || (measure(base) === 1 && !cvc(base))) {
      stem = base;
    }
  }

  // Step 5b
  if (measure(stem) > 1 && endsWithDoubleConsonant(stem) && stem.endsWith('l')) {
    stem = stem.slice(0, -1);
  }

  return stem;
}