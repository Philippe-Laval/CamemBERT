using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace CamemBERT.Library.Tests;

[TestClass]
public sealed class WordPieceTokenizerFactoryTest
{
    [TestMethod]
    public void TestWordPieceTokenizerFactory()
    {
        var tokenizer = WordPieceTokenizerFactory.Create();

        string text = "tokenization is fun.";
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

        Console.WriteLine($"Text:   \"{text}\"");
        Console.WriteLine($"Tokens: {ids.Count}");
        Console.WriteLine($"IDs:    [{string.Join(", ", ids)}]");

        // Detailed tokens
        Console.WriteLine("Detailed tokens");

        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
        foreach (EncodedToken token in tokens)
        {
            Console.WriteLine($"  '{token.Value,-15}' → ID {token.Id,6}");
        }

        // Decode
        Console.WriteLine("Decode");

        string? decoded = tokenizer.Decode(ids);
        Console.WriteLine($"Decoded: \"{decoded}\"");
    }

    [TestMethod]
    public void TestEncodeToIdsWithUNK()
    {
        var tokenizer = WordPieceTokenizerFactory.Create();

        // If "xyzzy" has no matching subwords in vocab:
        string text = "xyzzy";
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

        // Result: [UNK]
        // The ENTIRE word is replaced — not individual characters

        // Compare with BPE, which would fall back to:
        // ["x", "y", "z", "z", "y"] or byte-level encoding

        Console.WriteLine($"Text:   \"{text}\"");
        Console.WriteLine($"Tokens: {ids.Count}");
        Console.WriteLine($"IDs:    [{string.Join(", ", ids)}]");

        // Detailed tokens
        Console.WriteLine("Detailed tokens");

        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
        foreach (EncodedToken token in tokens)
        {
            Console.WriteLine($"  '{token.Value,-15}' → ID {token.Id,6}");
        }

        // Decode
        Console.WriteLine("Decode");

        string? decoded = tokenizer.Decode(ids);
        Console.WriteLine($"Decoded: \"{decoded}\"");
    }
}
