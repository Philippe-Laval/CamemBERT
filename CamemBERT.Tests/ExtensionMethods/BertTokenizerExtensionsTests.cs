using CamemBERT.ExtensionMethods;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT.Tests.ExtensionMethods
{
    [TestClass]
    public sealed class BertTokenizerExtensionsTests
    {
        [TestMethod]
        public void TestPreprocessing_SmallCaps()
        {
            // BERT embedding preprocessing pipeline:
             BertTokenizer tokenizer = BertTokenizerFactory.Create(
                new BertOptions { LowerCaseBeforeTokenization = true });

            string[] sentences = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing enables text understanding.",
                "Computer vision allows machines to interpret images.",
                "Tokenization converts text into numerical representations.",
                "AI is transforming healthcare, finance, and education.",
            ];

            var (inputIds, attentionMasks) = tokenizer.Preprocessing(sentences, 10);

            // Show first sentence as example
            Console.WriteLine("  Example (sentence 0):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[0].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[0].Take(20))}...]");
            Console.WriteLine();

            // Show last sentence as example
            Console.WriteLine("  Example (sentence 5):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[5].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[5].Take(20))}...]");
            Console.WriteLine();
        }

        [TestMethod]
        public void TestPreprocessing_BigCaps()
        {
            // BERT embedding preprocessing pipeline:
            BertTokenizer tokenizer = BertTokenizerFactory.Create(
               new BertOptions { LowerCaseBeforeTokenization = true });

            string[] sentences = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing enables text understanding.",
                "Computer vision allows machines to interpret images.",
                "Tokenization converts text into numerical representations.",
                "AI is transforming healthcare, finance, and education.",
            ];

            var (inputIds, attentionMasks) = tokenizer.Preprocessing(sentences, 128);

            // Show first sentence as example
            Console.WriteLine("  Example (sentence 0):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[0].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[0].Take(20))}...]");
            Console.WriteLine();

            // Show last sentence as example
            Console.WriteLine("  Example (sentence 5):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[5].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[5].Take(20))}...]");
            Console.WriteLine();
        }

        [TestMethod]
        public void TestPreprocessing_NoCaps()
        {
            // BERT embedding preprocessing pipeline:
            BertTokenizer tokenizer = BertTokenizerFactory.Create(
               new BertOptions { LowerCaseBeforeTokenization = true });

            string[] sentences = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing enables text understanding.",
                "Computer vision allows machines to interpret images.",
                "Tokenization converts text into numerical representations.",
                "AI is transforming healthcare and education.",
            ];

            var (inputIds, attentionMasks) = tokenizer.Preprocessing(sentences);

            // Show first sentence as example
            Console.WriteLine("  Example (sentence 0):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[0].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[0].Take(20))}...]");
            Console.WriteLine();

            // Show last sentence as example
            Console.WriteLine("  Example (sentence 5):");
            Console.WriteLine($"    input_ids:      [{string.Join(", ", inputIds[5].Take(20))}...]");
            Console.WriteLine($"    attention_mask: [{string.Join(", ", attentionMasks[5].Take(20))}...]");
            Console.WriteLine();
        }
    }
}
