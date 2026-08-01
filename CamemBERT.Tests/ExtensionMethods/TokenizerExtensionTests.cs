using CamemBERT.ExtensionMethods;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests.ExtensionMethods;

[TestClass]
public sealed class TokenizerExtensionTests
{
    [TestMethod]
    public void TestChunkWithOverlap()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();

        // Simulate a long document (article about AI)
        string document = """
Artificial Intelligence: A Comprehensive Overview

Introduction

Artificial intelligence (AI) is a branch of computer science that aims to create
intelligent machines that can perform tasks that typically require human intelligence.
These tasks include learning, reasoning, problem-solving, perception, and language
understanding. AI has become an increasingly important part of modern technology,
with applications ranging from virtual assistants to autonomous vehicles.

History of AI

The concept of artificial intelligence dates back to ancient myths and stories about
artificial beings endowed with intelligence. The modern field of AI research was
founded at a workshop at Dartmouth College in 1956. The attendees, including John
McCarthy, Marvin Minsky, Allen Newell, and Herbert Simon, became the leaders of AI
research for decades. They and their students produced programs that the press
described as "astonishing": computers were learning checkers strategies, solving
word problems in algebra, proving logical theorems, and speaking English.

By the middle of the 1960s, research in the United States was heavily funded by the
Department of Defense. Optimism was high, and researchers predicted that a machine
as intelligent as a human would exist within a generation. However, they were unable
to deliver on their promises, leading to what became known as the "AI winter" — a
period of reduced funding and interest in AI research that lasted from the mid-1970s
through the 1980s.

Machine Learning

Machine learning is a subset of AI that provides systems the ability to automatically
learn and improve from experience without being explicitly programmed. It focuses on
the development of computer programs that can access data and use it to learn for
themselves. The process of learning begins with observations or data, such as examples,
direct experience, or instruction, in order to look for patterns in data and make
better decisions in the future.

The primary aim is to allow the computers to learn automatically without human
intervention or assistance and adjust actions accordingly. Machine learning algorithms
are typically classified into three categories: supervised learning, unsupervised
learning, and reinforcement learning.

Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks
with multiple layers (hence "deep") to model and process complex patterns in data.
Deep learning has been revolutionary in many fields, including computer vision,
natural language processing, and speech recognition. Key architectures include
Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks
(RNNs) for sequential data, and Transformers for natural language processing.

The transformer architecture, introduced in the paper "Attention is All You Need"
by Vaswani et al. in 2017, has become the foundation for most modern large language
models. Transformers use self-attention mechanisms to process input sequences in
parallel, making them significantly faster to train than previous architectures.

Large Language Models

Large language models (LLMs) are deep learning models trained on vast amounts of text
data. They can generate human-like text, translate languages, write different kinds of
creative content, and answer questions in an informative way. Notable examples include
OpenAI's GPT series, Google's Gemini, Meta's Llama, and Anthropic's Claude.

These models use tokenization as their first processing step — converting raw text into
sequences of tokens that the model can process. Understanding tokenization is crucial
for working effectively with LLMs, as it affects everything from prompt engineering
to cost estimation.

Ethical Considerations

As AI becomes more powerful and ubiquitous, ethical considerations become increasingly
important. Key concerns include bias in AI systems, privacy implications, job
displacement, autonomous weapons, and the long-term existential risk posed by
superintelligent AI. Researchers and policymakers are working to develop guidelines
and regulations to ensure AI is developed and deployed responsibly.
""";

        int totalTokens = tokenizer.CountTokens(document);
        Console.WriteLine($"Document length: {document.Length} characters, {totalTokens} tokens\n");

        // ── Step 2: Configure chunking ──────────────────────────────────────────────
        Console.WriteLine("Step 2: Configure chunking parameters\n");

        int chunkSize = 100;     // tokens per chunk
        int overlapTokens = 20;  // overlap between chunks

        Console.WriteLine($"  Chunk size: {chunkSize} tokens");
        Console.WriteLine($"  Overlap:    {overlapTokens} tokens");
        Console.WriteLine($"  Expected chunks: ~{(int)Math.Ceiling((double)totalTokens / (chunkSize - overlapTokens))}\n");

        // ── Step 3: Chunk the document ──────────────────────────────────────────────
        Console.WriteLine("Step 3: Chunk the document\n");

        var chunks = tokenizer.ChunkWithOverlap(document, chunkSize, overlapTokens);

        Console.WriteLine($"  {"Chunk",-8} {"Tokens",7} {"Chars",7}  Preview");
        Console.WriteLine($"  {new string('-', 8)} {new string('-', 7)} {new string('-', 7)}  {new string('-', 50)}");

        foreach (var (chunk, index) in chunks.Select((c, i) => (c, i)))
        {
            int chunkTokenCount = tokenizer.CountTokens(chunk);
            string preview = chunk.Trim().Replace("\n", " ");
            if (preview.Length > 50) preview = preview[..47] + "...";
            Console.WriteLine($"  Chunk {index + 1,-3} {chunkTokenCount,7} {chunk.Length,7}  {preview}");
        }

        // ── Step 4: Verify coverage ─────────────────────────────────────────────────
        Console.WriteLine($"\nStep 4: Verify coverage\n");

        // Check that all content is covered
        int totalChunkTokens = chunks.Sum(c => tokenizer.CountTokens(c));
        Console.WriteLine($"  Total document tokens: {totalTokens}");
        Console.WriteLine($"  Sum of chunk tokens:   {totalChunkTokens} (includes overlap)");
        Console.WriteLine($"  Overlap tokens:        ~{totalChunkTokens - totalTokens}");

        // ── Step 5: Prepare for embedding ───────────────────────────────────────────
        Console.WriteLine($"\nStep 5: Prepare chunks for embedding API\n");

        // Simulate preparing chunks for an embedding API with a 8192 token limit
        int embeddingLimit = 8192;

        foreach (var (chunk, index) in chunks.Select((c, i) => (c, i)))
        {
            int tokens = tokenizer.CountTokens(chunk);
            bool fits = tokens <= embeddingLimit;
            Console.WriteLine($"  Chunk {index + 1}: {tokens,4} tokens — {(fits ? "✅ fits" : "❌ too large")} (limit: {embeddingLimit})");
        }

        Console.WriteLine("\n  All chunks are ready for embedding! In a real RAG pipeline,");
        Console.WriteLine("  each chunk would be sent to an embedding model, stored in a");
        Console.WriteLine("  vector database, and retrieved based on query similarity.");

        Console.WriteLine("\n✅ 04-text-chunking-pipeline.cs completed successfully");

    }
}
