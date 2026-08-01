using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests;

[TestClass]
public sealed class BertTokenizerFactoryTests
{
    [TestMethod]
    public void TestBertTokenizerFactory()
    {
        var tokenizer = BertTokenizerFactory.Create();

        string text = "Hello, world!";
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
    public void TestEncodeSentencePair()
    {
        var tokenizer = BertTokenizerFactory.Create();

        string sentenceA = "What is the capital of France?";
        string sentenceB = "The capital of France is Paris.";

        var (pairedIds, typeIds) = BertTokenizerFactory.EncodeSentencePair(tokenizer, sentenceA, sentenceB);

        Assert.AreEqual(pairedIds.Count, typeIds.Count);
    }

    [TestMethod]
    public void TestEncodeToIdsWithUNK()
    {
        var tokenizer = BertTokenizerFactory.Create();

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
