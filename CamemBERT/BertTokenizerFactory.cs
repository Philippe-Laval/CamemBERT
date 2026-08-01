using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/05-wordpiece-bert-tokenizer.md

/*
Sentence-Pair Encoding

BERT supports paired inputs for tasks like question answering, natural language inference, and sentence similarity. The format is:

[CLS] sentence_A [SEP] sentence_B [SEP]

Each token gets a token type ID indicating which sentence it belongs to:

Tokens:    [CLS]  Hello  ,  world  [SEP]  How  are  you  ?  [SEP]
Type IDs:    0      0    0    0      0      1    1    1   1    1
             └─── Sentence A ───┘          └── Sentence B ──┘
 
 */


namespace CamemBERT
{
    public class BertTokenizerFactory
    {
        public static BertTokenizer Create()
        {
            using var modelStream = GetBertBaseUncasedVocabStream();

            if (modelStream == null)
                throw new InvalidOperationException("vocab.txt stream is null");

            BertTokenizer tokenizer = BertTokenizer.Create(modelStream,
                new BertOptions
                {
                    // Important: BERT-base-uncased is case-insensitive, so we lowercase before tokenization
                    LowerCaseBeforeTokenization = true,
                    ClassificationToken = "[CLS]",
                    SeparatorToken = "[SEP]",
                    PaddingToken = "[PAD]"
                });

            return tokenizer;
        }

        /// <summary>
        /// Encode a sentence pair
        /// </summary>
        /// <param name="sentenceA"></param>
        /// <param name="sentenceB"></param>
        public static (List<int> PairedIds, List<int> TypeIds) EncodeSentencePair(BertTokenizer tokenizer, string sentenceA, string sentenceB)
        {
            // Encode each sentence
            var idsA = tokenizer.EncodeToIds(sentenceA);
            var idsB = tokenizer.EncodeToIds(sentenceB);

            // Build paired input manually:
            // [CLS] + A_tokens + [SEP] + B_tokens + [SEP]
            var pairedIds = new List<int>();
            pairedIds.Add(101);              // [CLS]
            pairedIds.AddRange(idsA);        // Sentence A (may already include CLS/SEP from BertTokenizer)
            pairedIds.Add(102);              // [SEP]
                                             // For sentence B, encode without CLS/SEP
            pairedIds.AddRange(idsB.Skip(1).SkipLast(1));  // Strip CLS and SEP from B
            pairedIds.Add(102);              // [SEP]

            // Token type IDs
            var typeIds = new List<int>();
            typeIds.AddRange(Enumerable.Repeat(0, idsA.Count + 2));   // A + CLS + SEP
            typeIds.AddRange(Enumerable.Repeat(1, idsB.Count - 1));    // B + SEP

            return (pairedIds, typeIds);
        }

        /*
         For batch processing, sequences must be the same length. Use padding:

int maxLength = 128;  // or max length in batch

int[] inputIds = new int[maxLength];
int[] attentionMask = new int[maxLength];
int[] tokenTypeIds = new int[maxLength];

var ids = tokenizer.EncodeToIds(text);

// Copy actual tokens
for (int i = 0; i < ids.Count && i < maxLength; i++)
{
    inputIds[i] = ids[i];
    attentionMask[i] = 1;   // 1 = real token
}
        Confusing token type IDs with attention masks. Token type IDs (0/1) indicate sentence A vs B. Attention masks (0/1) indicate real tokens vs padding. They serve different purposes.

         */


        // Charge le contenu embarqué de BertBaseUncased\vocab.txt comme flux.
        public static Stream? GetBertBaseUncasedVocabStream()
        {
            // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
            // Dans ce projet, le root namespace est "CamemBERT" et le fichier est BertBaseUncased\vocab.txt
            string resourceName = "CamemBERT.BertBaseUncased.vocab.txt";
            Assembly asm = Assembly.GetExecutingAssembly();
            Stream? stream = asm.GetManifestResourceStream(resourceName);
            return stream;
        }
    }
}
