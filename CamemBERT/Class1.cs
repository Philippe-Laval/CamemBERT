using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

// https://learn.microsoft.com/fr-fr/dotnet/ai/how-to/use-tokenizers

namespace CamemBERT
{
    internal class Class1
    {
        /// <summary>
        /// GPT-5 tokenizer
        /// </summary>
        public void Test1()
        {
            // Initialize the tokenizer for the gpt-5 model.
            Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-5");

            string source = "Text tokenization is the process of splitting a string into a list of tokens.";

            // Count the tokens in the text.
            Console.WriteLine($"Tokens: {tokenizer.CountTokens(source)}");
            // Output: Tokens: 16

            // Encode text to token IDs.
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(source);
            Console.WriteLine($"Token IDs: {string.Join(", ", ids)}");
            // Output: Token IDs: 1279, 6602, 2860, 382, 290, 2273, 328, 87130, 261, 1621, 1511, 261, 1562, 328, 20290, 13

            // Decode token IDs back to text.
            string? decoded = tokenizer.Decode(ids);
            Console.WriteLine($"Decoded: {decoded}");
            // Output: Decoded: Text tokenization is the process of splitting a string into a list of tokens.
        }


        public void Test2()
        {
            Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-5");

            string source = "Text tokenization is the process of splitting a string into a list of tokens.";

            // Get the last 5 tokens from the text.
            var trimIndex = tokenizer.GetIndexByTokenCountFromEnd(source, 5, out string? processedText, out _);
            processedText ??= source;
            Console.WriteLine($"Last 5 tokens: {processedText.Substring(trimIndex)}");
            // Output: Last 5 tokens:  a list of tokens.

            // Get the first 5 tokens from the text.
            trimIndex = tokenizer.GetIndexByTokenCount(source, 5, out processedText, out _);
            processedText ??= source;
            Console.WriteLine($"First 5 tokens: {processedText.Substring(0, trimIndex)}");
            // Output: First 5 tokens: Text tokenization is the
        }

        /// <summary>
        /// Llama tokenizer
        /// </summary>
        /// <returns></returns>
        public async Task Test3()
        {
            // Open a stream to the remote Llama tokenizer model data file.
            using HttpClient httpClient = new();
            const string modelUrl = @"https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model";
            using Stream remoteStream = await httpClient.GetStreamAsync(modelUrl);

            // Create the Llama tokenizer using the remote stream.
            Tokenizer llamaTokenizer = LlamaTokenizer.Create(remoteStream);

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

        /// <summary>
        /// BPE (Byte Pair Encoding) tokenizer
        /// </summary>
        /// <returns></returns>
        public async Task Test4()
        {
            // BPE (Byte Pair Encoding) tokenizer can be created from vocabulary and merges files.
            // Download the GPT-2 tokenizer files from Hugging Face.
            using HttpClient httpClient = new();
            const string vocabUrl = @"https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
            const string mergesUrl = @"https://huggingface.co/openai-community/gpt2/raw/main/merges.txt";

            using Stream vocabStream = await httpClient.GetStreamAsync(vocabUrl);
            using Stream mergesStream = await httpClient.GetStreamAsync(mergesUrl);

            // Create the BPE tokenizer using the vocabulary and merges streams.
            Tokenizer bpeTokenizer = BpeTokenizer.Create(vocabStream, mergesStream);

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
}
