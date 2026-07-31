using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests
{
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
    }
}
