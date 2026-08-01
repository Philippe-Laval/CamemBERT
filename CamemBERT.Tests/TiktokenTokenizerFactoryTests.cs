using Microsoft.ML.Tokenizers;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/06-encoding-decoding.md

namespace CamemBERT.Tests
{
    [TestClass]
    public sealed class TiktokenTokenizerFactoryTests
    {
        [TestMethod]
        public void TestEncodeToIds_Basic()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "The quick brown fox jumps over the lazy dog.";

            // String overload
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);
            Console.WriteLine($"Token count: {ids.Count}");              // Token count: 10
            Console.WriteLine($"IDs: [{string.Join(", ", ids)}]");

            // Span overload (avoids string allocation if you already have a span)
            ReadOnlySpan<char> span = text.AsSpan();
            IReadOnlyList<int> ids2 = tokenizer.EncodeToIds(span);
        }

        [TestMethod]
        public void TestEncodeToIds_WithMaxTokenLimit()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "The quick brown fox jumps over the lazy dog.";

            // Encode at most 5 tokens, find out how many characters were consumed
            IReadOnlyList<int> truncatedIds = tokenizer.EncodeToIds(
            text,
            maxTokenCount: 5,
            out string? normalizedText,
            out int charsConsumed);

            Console.WriteLine($"Truncated to {truncatedIds.Count} tokens");   // 5
            Console.WriteLine($"Characters consumed: {charsConsumed}");        // ~23
            Console.WriteLine($"Text consumed: '{text[..charsConsumed]}'");
        }

        [TestMethod]
        public void TestEncodeToToken_Detailed_Token_Information()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "The quick brown fox jumps over the lazy dog.";

            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out string? normalized);

            foreach (EncodedToken token in tokens)
            {
                string sourceText = text[token.Offset];  // Extract the original text this token came from
                Console.WriteLine($"  Token: '{token.Value}' | ID: {token.Id} | " +
                              $"Offset: [{token.Offset.Start}..{token.Offset.End}] | " +
                              $"Source: '{sourceText}'");
            }
        }


        [TestMethod]
        public void TestCountTokens()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "The quick brown fox jumps over the lazy dog.";

            // When you only need the count, CountTokens is more efficient
            // (doesn't allocate the token ID list)
            int count = tokenizer.CountTokens(text);
            Console.WriteLine($"Token count: {count}");  // Token count: 10

            // Equivalent but less efficient:
            int countAlt = tokenizer.EncodeToIds(text).Count;  // Allocates the list
        }

        [TestMethod]
        public void TestDecode()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "Hello, world!";

            // Encode
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

            // Decode
            string? decoded = tokenizer.Decode(ids);
            Console.WriteLine($"Decoded: '{decoded}'");  // Decoded: 'Hello, world!'

            // Decode arbitrary IDs (e.g., from model output)
            string? fromModel = tokenizer.Decode(new[] { 15339, 11, 1917, 0 });
            Console.WriteLine($"From model: '{fromModel}'");  // From model: ' awesome, down!'
        }


        [TestMethod]
        public void TestTokenOffsets()
        {
            /*
    EncodedToken.Offset is a Range that maps each token back to its position in the original (or normalized) text. This is invaluable for:

    Highlighting which part of the text each token came from
    Named Entity Recognition — mapping token-level labels back to text spans
    Debugging — understanding token boundaries  
             */

            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "Microsoft's AI tokenizer";
            var tokens = tokenizer.EncodeToTokens(text, out _);

            foreach (var token in tokens)
            {
                int start = token.Offset.Start.Value;
                int end = token.Offset.End.Value;
                string highlight = text[start..end];
                Console.WriteLine($"  [{start,2}..{end,2}] '{token.Value}' → '{highlight}'");
            }

        }

        [TestMethod]
        public void TestBypassingPipelineStages()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.Create();
            string text = "The quick brown fox jumps over the lazy dog.";

            // Skip normalization (text already preprocessed)
            var ids = tokenizer.EncodeToIds(text,
                considerNormalization: false);

            // Skip both normalization and pre-tokenization
            var ids2 = tokenizer.EncodeToIds(text,
                considerNormalization: false,
                considerPreTokenization: false);
        }


    }
}
