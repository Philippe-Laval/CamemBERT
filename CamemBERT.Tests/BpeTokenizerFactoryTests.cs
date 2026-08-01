using Microsoft.ML.Tokenizers;
using static System.Net.Mime.MediaTypeNames;

namespace CamemBERT.Library.Tests;

[TestClass]
public sealed class BpeTokenizerFactoryTests
{
    [TestMethod]
    public void TestBpeTokenizerFactoryCreate()
    {
        BpeTokenizer bpeTokenizer = BpeTokenizerFactory.Create();

        // Encode
        Console.WriteLine("\n═══ 3. Encode and decode ═══");

        string text = "Hello, world!";
        IReadOnlyList<int> ids = bpeTokenizer.EncodeToIds(text);

        Console.WriteLine($"Text:    \"{text}\"");
        Console.WriteLine($"Tokens:  {ids.Count}");
        Console.WriteLine($"IDs:     [{string.Join(", ", ids)}]");

        // Detailed tokens
        IReadOnlyList<EncodedToken> tokens = bpeTokenizer.EncodeToTokens(text, out _);
        foreach (EncodedToken token in tokens)
        {
            Console.WriteLine($"  '{token.Value}' → ID {token.Id}");
        }

        // Decode
        string? decoded = bpeTokenizer.Decode(ids);
        Console.WriteLine($"Decoded: \"{decoded}\"");
    }

    [TestMethod]
    public void TestBpeTokenizerFactoryCreateWithCustomOptions()
    {
        // https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/cookbook/bpe-with-options.cs
        // Not working here since GPT2 has no special 

        BpeTokenizer bpeTokenizer = BpeTokenizerFactory.CreateWithCustomOptions();

        // Encode with special tokens
        Console.WriteLine("\n═══ 3. Encode text ═══");

        string text = "Hello, world!";
        IReadOnlyList<int> ids = bpeTokenizer.EncodeToIds(text);
        Console.WriteLine($"Text: \"{text}\"");
        Console.WriteLine($"IDs:  [{string.Join(", ", ids)}]");

        IReadOnlyList<EncodedToken> tokens = bpeTokenizer.EncodeToTokens(text, out _);
        foreach (var token in tokens)
        {
            Console.WriteLine($"  '{token.Value}' → ID {token.Id}");
        }

        // Unknown token behavior
        Console.WriteLine("\n═══ 4. Unknown token behavior ═══");

        // Characters not in vocab will map to unknown token
        string unknownText = "xyz";
        IReadOnlyList<int> unknownIds = bpeTokenizer.EncodeToIds(unknownText);
        Console.WriteLine($"Text: \"{unknownText}\"");
        Console.WriteLine($"IDs:  [{string.Join(", ", unknownIds)}]");
        Console.WriteLine("Characters not in vocab → <unk> token");
    }

    [TestMethod]
    public void TestBpeTokenizerFactoryCreateWithLowerCaseNormalizer()
    {
        BpeTokenizer lowerTokenizer = BpeTokenizerFactory.Create(new LowerCaseNormalizer());
        
        string mixedCase = "Hello, World!";
        IReadOnlyList<int> lowerIds = lowerTokenizer.EncodeToIds(mixedCase);
        Console.WriteLine($"Input:      \"{mixedCase}\"");
        Console.WriteLine($"Normalized: (lowercased by LowerCaseNormalizer)");
        Console.WriteLine($"IDs:        [{string.Join(", ", lowerIds)}]");

        // Detailed tokens
        IReadOnlyList<EncodedToken> tokens = lowerTokenizer.EncodeToTokens(mixedCase, out _);
        foreach (EncodedToken token in tokens)
        {
            Console.WriteLine($"  '{token.Value}' → ID {token.Id}");
        }

        // Decode
        string? decoded = lowerTokenizer.Decode(lowerIds);
        Console.WriteLine($"Decoded: \"{decoded}\"");
    }


    [TestMethod]
    public async Task Test4()
    {
        // BPE (Byte Pair Encoding) tokenizer can be created from vocabulary and merges files.
        // Download the GPT-2 tokenizer files from Hugging Face.
        //using HttpClient httpClient = new();
        //const string vocabUrl = @"https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
        //const string mergesUrl = @"https://huggingface.co/openai-community/gpt2/raw/main/merges.txt";

        //using Stream vocabStream = await httpClient.GetStreamAsync(vocabUrl);
        //using Stream mergesStream = await httpClient.GetStreamAsync(mergesUrl);

        // Create the BPE tokenizer using the vocabulary and merges streams.
        Tokenizer bpeTokenizer = BpeTokenizerFactory.Create();

        string text = "Hello, how are you doing today?";

        // Encode text to token IDs.
        IReadOnlyList<int> ids = bpeTokenizer.EncodeToIds(text);
        Console.WriteLine($"Token IDs: {string.Join(", ", ids)}");

        // Count tokens.
        int tokenCount = bpeTokenizer.CountTokens(text);
        Console.WriteLine($"Token count: {tokenCount}");

        // Get detailed token information.
        IReadOnlyList<EncodedToken> tokens = bpeTokenizer.EncodeToTokens(text, out string? normalizedString);
        Console.WriteLine("Tokens:");
        foreach (EncodedToken token in tokens)
        {
            Console.WriteLine($"  ID: {token.Id}, Value: '{token.Value}'");
        }

        // Decode tokens back to text.
        string? decoded = bpeTokenizer.Decode(ids);
        Console.WriteLine($"Decoded: {decoded}");

        // Note: BpeTokenizer might not always decode IDs to the exact original text
        // as it can remove spaces during tokenization depending on the model configuration.
    }

}
