using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests
{
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
    }
}
