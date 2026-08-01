using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests;

[TestClass]
public sealed class LlamaTokenizerFactoryTests
{
    [TestMethod]
    public void TestLlamaTokenizerFactoryCreate()
    {
        LlamaTokenizer tokenizer = LlamaTokenizerFactory.Create();

        string text = "Hello, world! How are you?";

        // Encode
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);
        Console.WriteLine($"Token count: {ids.Count}");
        Console.WriteLine($"IDs: [{string.Join(", ", ids)}]");

        // Detailed tokens (shows ▁ markers)
        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
        foreach (var token in tokens)
        {
            Console.WriteLine($"  '{token.Value}' → {token.Id}");
        }

        // Decode
        string decoded = tokenizer.Decode(ids)!;
        Console.WriteLine($"Decoded: {decoded}");
    }


    [TestMethod]
    public async Task Test3()
    {
        // Create the Llama tokenizer using the remote stream.
        Tokenizer llamaTokenizer = LlamaTokenizerFactory.Create();

        string input = "Hello, world!";

        // Encode text to token IDs.
        IReadOnlyList<int> ids = llamaTokenizer.EncodeToIds(input);
        Console.WriteLine($"Token IDs: {string.Join(", ", ids)}");
        // Output: Token IDs: 1, 15043, 29892, 3186, 29991

        // Count the tokens.
        Console.WriteLine($"Tokens: {llamaTokenizer.CountTokens(input)}");
        // Output: Tokens: 5

        // Decode token IDs back to text.
        string? decoded = llamaTokenizer.Decode(ids);
        Console.WriteLine($"Decoded: {decoded}");
        // Output: Decoded: Hello, world!


        // Tous les tokenizers prennent en charge les options d’encodage avancées, telles que le contrôle de la normalisation et la prétokenisation 

        ReadOnlySpan<char> textSpan = "Hello World".AsSpan();

        // Bypass normalization during encoding.
        ids = llamaTokenizer.EncodeToIds(textSpan, considerNormalization: false);

        // Bypass pretokenization during encoding.
        ids = llamaTokenizer.EncodeToIds(textSpan, considerPreTokenization: false);

        // Bypass both normalization and pretokenization.
        ids = llamaTokenizer.EncodeToIds(textSpan, considerNormalization: false, considerPreTokenization: false);
    }
}
