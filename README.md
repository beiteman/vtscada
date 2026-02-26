# README

## Getting Started

Follow these steps to get the project up and running.

### Requirement

Make sure these components installed:

| Requirement | Recommended Version |
|-------------|---------|
| **Windows** | 11 |
| **Node.js** | v20.19.4 |
| **VS Code** | 1.103.2 |

> **Note:** Recommended versions are based on successful testing. Other versions may also work.

## Compile Instruction

1. Navigate to the project folder
2. Run `./compile.bat`

> If successfull, this step producing file `vtscada-function-retrieval-<version>.vsix`

## Testing

1. Run `npm install`
2. Run `npx tsx test.ts`

> If successfull, this step producing file `tests/test-results.txt`

## Install in the VS Code

1. In Visual Studio Code, click the **Extensions** icon in the Activity Bar
2. Click the **More Actions** button (`...`) in the top-right corner of the Extensions view
3. Select **Install from VSIX...**
4. Browse and select the VSIX file `vtscada-function-retrieval-<version>.vsix`
5. Click **Install**

> After installation, user can create a file with the `.SRC` extension to start using the extension.

## Documentation

### 1. Overview

This project implements a **function/document retrieval system for VTScada**, packaged as a **VS Code extension**. Its primary goal is to help developers quickly retrieve relevant VTScada functions or documentation snippets based on a natural-language query.

The system is **embedding-based**, using Transformer models (ONNX) to encode both documentation and user queries into vector representations, then performing similarity search to find the most relevant results.

The project has evolved over time:

* Early versions used **BM25** and **TF-IDF** (now deprecated)
* The current production system uses **Transformer-based sentence embeddings**

This document is intended to allow a **new developer to fully understand, operate, and extend the system** without prior context.

---

### 2. High-Level Architecture

At a high level, the system consists of four major layers:

1. **Documentation Data Layer**
   Multilingual VTScada documentation stored as structured JSON files.

2. **Indexing Layer (Offline)**
   Converts documentation into vector indexes using Transformer models.

3. **Retrieval Layer (Runtime)**
   Encodes user queries and retrieves the most relevant documentation entries.

4. **VS Code Extension Layer**
   Integrates retrieval into the editor experience.

```
┌──────────────────────────┐
│   VS Code Extension UI   │
│  (extension.ts)         │
└────────────┬─────────────┘
             │ Query
┌────────────▼─────────────┐
│   Retrieval Engine       │
│  (transformer.ts)       │
└────────────┬─────────────┘
             │ Embeddings
┌────────────▼─────────────┐
│  Language Detection     │
│  (languages.ts)         │
└────────────┬─────────────┘
             │ Language
┌────────────▼─────────────┐
│  Vector Index Files     │
│  (resource/index.*.json)│
└──────────────────────────┘
```

---

### 3. Repository Structure

```
compile.bat         # Script to download models, build index and compile as vsix file
build.ts            # Builds vector indexes from documentation
test.ts             # Test runner for retrieval logic
/tests
    testcases.json  # Input queries and expected result
/docs
    docs.en.json    # English VTScada documentation
    docs.zh-cn.json # Simplified Chinese documentation
    docs.zh-tw.json # Traditional Chinese documentation
/resource
    /all-MiniLM-L6-v2
        model.onnx
        tokenizer.json
        tokenizer_config.json
    /text2vec-base-chinese
        model.onnx
        tokenizer.json
        tokenizer_config.json
    /index.en.json
    /index.zh-cn.json
    /index.zh-tw.json
/src
    bm25.ts         # Legacy BM25 retrieval (unused)
    tfidf.ts        # Legacy TF-IDF retrieval (unused)
    transformer.ts  # Current Transformer-based retrieval
    extension.ts    # VS Code extension entry point
    languages.ts    # Query language detection
    utils.ts        # Shared helper utilities
```

---

### 4. Documentation Data Format

Documentation is stored as JSON files under `/docs`.

Each language has its own file:

* `docs.en.json`
* `docs.zh-cn.json`
* `docs.zh-tw.json`

#### Expected Structure (Conceptual)

Each documentation entry contains:

* Function name
* Description / Comments
* Code Snippets

These entries are treated as **atomic retrieval units** during indexing using the "comments".

---

### 5. Models and Resources

The system uses **ONNX-based Transformer models** for sentence embedding.

#### English Model

* **Model**: `all-MiniLM-L6-v2`
* Source: HuggingFace sentence-transformers
* Used for English documentation and queries

#### Chinese Model

* **Model**: `text2vec-base-chinese`
* Source: HuggingFace (shibing624)
* Used for Simplified and Traditional Chinese

#### Required Files per Model

Each model directory must contain:

* `model.onnx`
* `tokenizer.json`
* `tokenizer_config.json`

These files are loaded at runtime for embedding generation.

---

### 6. Language Detection

File: `src/languages.ts`

Purpose:

* Detects the language of the user's query
* Routes the query to the correct model and index

Typical behavior:

* English → `index.en.json`
* Simplified Chinese → `index.zh-cn.json`
* Traditional Chinese → `index.zh-tw.json`

This ensures semantic consistency between query and documentation embeddings.

---

### 7. Indexing Pipeline (Offline Build)

Indexing is performed **offline** and must be done before testing or packaging.

#### Step 1: Preparation

Download required model files:

**English model**

```
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

Place into:

```
resource/all-MiniLM-L6-v2/
```

**Chinese model**

```
https://huggingface.co/shibing624/text2vec-base-chinese
```

Place into:

```
resource/text2vec-base-chinese/
```

Then install dependencies:

```
npm install
```

---

#### Step 2: Build Indexes

Compile and run the build script:

```
npx tsx build.ts
```

This process:

1. Loads documentation JSON files
2. Generates embeddings for each entry
3. Writes vector indexes to disk

#### Output Files

```
resource/index.en.json
resource/index.zh-cn.json
resource/index.zh-tw.json
```

Each index contains:

* Original documentation metadata
* Precomputed embedding vectors

---

### 8. Retrieval Engine

File: `src/transformer.ts`

This is the **core runtime component**.

#### Responsibilities

* Load ONNX model and tokenizer
* Encode user query into a vector
* Load corresponding index file
* Compute similarity (e.g., cosine similarity)
* Rank results
* Return top-k matches

#### Notes

* Legacy files `bm25.ts` and `tfidf.ts` are kept for reference but are not used
* Any new retrieval strategy should follow the same interface as `transformer.ts`

---

### 9. Testing

Testing is done outside VS Code using a standalone runner.

#### Test Data

File:

```
/tests/testcases.json
```

Contains:

* Query strings
* Expected relevant functions or descriptions

#### Run Tests

```
npx tsx test.ts
```

#### Output

```
tests/test-result.txt
```

This file contains:

* Query
* Retrieved results
* Similarity scores

Used to manually evaluate retrieval quality.

---

### 10. VS Code Extension

File: `src/extension.ts`

This is the entry point for the VS Code extension.

#### Responsibilities

* Register VS Code commands
* Capture user input
* Call retrieval engine
* Display results inside VS Code

The extension does **not** build indexes — it only consumes prebuilt ones.

---

### 11. Building the VS Code Extension

#### Step 1: Set Version

Edit `package.json`:

```
"version": "x.y.z"
```

---

#### Step 2: Compile and Package

```
npm run compile
npx vsce package
```

#### Output

```
vtscada-function-retrieval-x.y.z.vsix
```

This file can be installed directly into VS Code.

---

### 12. Development Notes for Future Maintainers

#### Adding a New Language

1. Add documentation JSON under `/docs`
2. Add or select an appropriate embedding model
3. Extend `languages.ts`
4. Rebuild indexes

#### Changing the Retrieval Model

* Implement new logic in a separate file
* Keep interface compatible with `transformer.ts`
* Update `extension.ts` to switch engines if needed

#### Performance Considerations

* Index size grows linearly with documentation
* Embedding computation is the most expensive step
* Runtime retrieval is fast due to precomputed vectors

---
