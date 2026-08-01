using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests
{
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
            Console.WriteLine("\n═══ 3. Detailed tokens ═══");

            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
            foreach (EncodedToken token in tokens)
            {
                Console.WriteLine($"  '{token.Value,-15}' → ID {token.Id,6}");
            }

            // Decode
            Console.WriteLine("\n═══ 4. Decode ═══");

            string? decoded = tokenizer.Decode(ids);
            Console.WriteLine($"Decoded: \"{decoded}\"");
        }
    }
}
