using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace CamemBERT.Tests
{
    [TestClass]
    public sealed class Qwen3TokenizerFactoryTests
    {
        [TestMethod]
        public void TestQwen3TokenizerFactory()
        {
            var query = "Your query here";
            var task = $"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}";

            var tokenizer = Qwen3TokenizerFactory.Create();
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(task);

            Console.WriteLine($"Text:    \"{task}\"");
            Console.WriteLine($"Tokens:  {ids.Count}");
            Console.WriteLine($"IDs:     [{string.Join(", ", ids)}]");

            // Detailed tokens
            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(task, out _);
            foreach (EncodedToken token in tokens)
            {
                Console.WriteLine($"  '{token.Value}' → ID {token.Id}");
            }

            // Decode
            string? decoded = tokenizer.Decode(ids);
            Console.WriteLine($"Decoded: \"{decoded}\"");


        }
    }
}
